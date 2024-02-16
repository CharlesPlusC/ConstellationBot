import os
import time
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image
import numpy as np
# import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

from source.tools.constellation_tools import pull_constellation_TLEs
from source.tools.conversions import sphere_coords, TLE_time, ecef2latlong, eci2ecef_astropy
from source.tools.propagator import sgp4_prop_TLE
from source.visualisation_maker.constellation_info import fetch_sat_info, pos_vel_from_statevecs

cwd = os.getcwd()

constellations = ['oneweb', 'starlink', 'planet', 'swarm', 'spire', 'iridium']
animation_folder_path = cwd + '/images/constellation_anim'

def generate_state_gif(const):
    constellation_paths, _ = fetch_sat_info(const=const, format="statevecs", anim='latest_state')
    eci_pos, _, alts = pos_vel_from_statevecs(constellation_paths)

    cmap = cm.get_cmap('jet')
    norm = Normalize(vmin=min(alts), vmax=max(alts))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('xkcd:steel grey')
    ax.xaxis.pane.set_facecolor('xkcd:steel grey')
    ax.yaxis.pane.set_facecolor('xkcd:steel grey')
    ax.zaxis.pane.set_facecolor('xkcd:steel grey')

    # Setup for the ax
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks(np.arange(-8000, 8001, 2000))
    ax.set_yticks(np.arange(-8000, 8001, 2000))
    ax.set_zticks(np.arange(-8000, 8001, 2000))
    ax.set_xticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax.set_yticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax.set_zticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    fig.patch.set_facecolor('xkcd:steel grey')
    fig.suptitle('Latest Positions:'+ const +'\n Date: ' + str(time.strftime("%d/%m/%y")) +", "+ str(len(eci_pos))+' satellites', fontsize=16, y=0.95, x=0.5, color='black')

    earth_x, earth_y, earth_z = sphere_coords(6378.138, granularity=50)

    scatters = ax.scatter(*np.array(eci_pos).T, c=scalar_map.to_rgba(alts), s=2)
    wireframe = ax.plot_wireframe(earth_x, earth_y, earth_z, color='xkcd:grey', alpha=0.4)

    # Adding the colorbar
    cb = fig.colorbar(scalar_map, ax=ax, shrink=0.5, aspect=10, label='Altitude (km)')

    def update(i):
        ax.view_init(30, i)
        return scatters, wireframe

    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), blit=True)

    gif_folder = os.path.join('images/constellation_anim/gifs/', const)
    os.makedirs(gif_folder, exist_ok=True)
    
    ani.save(os.path.join(gif_folder, f'{const}_{time.strftime("%y_%m_%d")}_anim.gif'), writer='imagemagick', fps=10)


def plot_satellite_data(ax, const_ephemerides, norm_alts_all, random_seed=42):
    # Set the random seed to ensure reproducible results
    np.random.seed(random_seed)

    # Calculate the length of each ephemeris and create a list of random numbers
    len_ephem = [len(ephem) for ephem in const_ephemerides]
    rand_list = [np.random.randint(0, len_ephem[i]) for i in range(len(len_ephem))]

    index = 0
    for sat_idx, sat in enumerate(const_ephemerides):
        ephems = np.array([ephem[1] for ephem in sat])
        num_satellites = len(ephems)
        rand = rand_list[sat_idx]  

        # Plot the first point of each ephemeris
        ax.scatter(
            ephems[rand][0],
            ephems[rand][1],
            ephems[rand][2],
            color="xkcd:black",
            alpha=0.8,
            marker="o",
            s=0.5
        )

        for i in range(len(ephems) - 1):
            x_values, y_values, z_values = ephems[i:i + 2].T

            # Plot the line without markers
            ax.plot(
                x_values,
                y_values,
                z_values,
                color = cm.jet(norm_alts_all[(index + i) % len(norm_alts_all)]),
                linewidth=0.15,
                alpha=0.4
            )

        index += num_satellites

    return

def process_geom_data(const):
    # fetch data and return the data to be plotted
    constellation_path, constellation_img_paths = fetch_sat_info(const, format="TLE", anim="current_geometry")

    # Add a check for whether a gif already exists
    gif_folder = os.path.join('images/constellation_anim/gifs/', const)
    gif_file = os.path.join(gif_folder, f'geom_{const}_{time.strftime("%y_%m_%d")}.gif')
    if os.path.exists(gif_file):
        print(f"Gif file for {const} already exists. Skipping...")
        pass

    const_ephemerides = []
    individual_TLEs = []
    with open(constellation_path, 'r') as f:
        for line in f:
            individual_TLEs.append(line.strip())
            if len(individual_TLEs) == 2:
                tle_string = '\n'.join(individual_TLEs)
                TLE_jd = TLE_time(tle_string)
                TLE_jd_plusdt = TLE_jd + 120/1440
                const_ephemerides.append(sgp4_prop_TLE(TLE=tle_string, jd_start=TLE_jd, jd_end=TLE_jd_plusdt, dt=120, alt_series=False))
                individual_TLEs = []
    return const_ephemerides, constellation_img_paths

def create_geom_frame(az, const_ephemerides, constellation_img_paths, const, elev=0):
    print(f"Inside geom frame {az} for {const}...")
    print(f"Creating frame for {constellation_img_paths + str(az) + '.png'}...")
    fig = plt.figure(figsize=(6, 6), facecolor='xkcd:steel grey') 

    # First subplot
    ax1 = fig.add_subplot(111, projection='3d', auto_add_to_figure=False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.add_axes(ax1)
    ax1.elev = elev
    ax1.azim = az
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_zticks(np.arange(-8000, 8001, 2000))
    ax1.set_yticks(np.arange(-8000, 8001, 2000))
    ax1.set_xticks(np.arange(-8000, 8001, 2000))

    ax1.set_zticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax1.set_yticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax1.set_xticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')

    # Label axes
    ax1.set_xlabel('X-eci (km)', fontsize=10, color='black')
    ax1.set_ylabel('Y-eci (km)', fontsize=10, color='black')
    ax1.set_zlabel('Z-eci (km)', fontsize=10, color='black')

    ax1.set_facecolor('xkcd:steel grey')
    ax1.xaxis.pane.set_facecolor('xkcd:steel grey')
    ax1.yaxis.pane.set_facecolor('xkcd:steel grey')
    ax1.zaxis.pane.set_facecolor('xkcd:steel grey')

    alts_all = []
    for sat in const_ephemerides:
        ephems = np.array([ephem[1] for ephem in sat])
        alts_sat = np.linalg.norm(ephems, axis=1) - 6378.137
        alts_all.extend(alts_sat)

    min_alt_all = np.min(alts_all)
    max_alt_all = np.max(alts_all)
    norm_alts_all = (alts_all - min_alt_all) / (max_alt_all - min_alt_all)

    
    plot_satellite_data(ax1, const_ephemerides, norm_alts_all=norm_alts_all)

    cb_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4]) # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=min_alt_all, vmax=max_alt_all))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cb_ax)
    cbar.set_label('Altitude (km)', fontsize=12, color='black', labelpad=5)
    cbar.ax.tick_params(labelsize=9, color='black')

    fig.subplots_adjust(right=0.8)  # Adjust space to the right of the subplots
    print(f"Saving frame for {const} at azimuth {az} at location {constellation_img_paths + str(az) + '.png'}")
    fig.suptitle('Orbital Configuration:' +const +'\n Date:' + str(time.strftime("%d/%m/%y")) + ', ' + str(len(const_ephemerides))+' satellites', fontsize=16, y=0.95, x=0.5, color='black')
    # Save the figure
    fig.savefig(constellation_img_paths + str(az) + '.png', dpi=200, pad_inches=0.1)
    print(f"Saved frame for {const} at azimuth {az} at location {constellation_img_paths + str(az) + '.png'}")

    plt.close()

def create_frame(args):
    az, const, const_ephemerides, constellation_img_paths = args
    print(f"Creating frame for {const} at azimuth {az}...")
    create_geom_frame(az, const_ephemerides, constellation_img_paths, const)

# def generate_geom_gif(const):
#     # Process the geometric data for the constellation
#     const_ephemerides, _ = process_geom_data(const)
#     print(f"Processing geometric data for {const}...")
#     print(f"length of const ephemerides: {len(const_ephemerides)}")

#     # Set up the figure and axis
#     fig = plt.figure(figsize=(6, 6), facecolor='xkcd:steel grey')
#     ax = fig.add_subplot(111, projection='3d')

#     # Axis setup code
#     ax.set_box_aspect([1, 1, 1])
#     ax.set_zticks(np.arange(-8000, 8001, 2000))
#     ax.set_yticks(np.arange(-8000, 8001, 2000))
#     ax.set_xticks(np.arange(-8000, 8001, 2000))
#     ax.set_xticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
#     ax.set_yticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
#     ax.set_zticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
#     ax.set_facecolor('xkcd:steel grey')
#     ax.xaxis.pane.set_facecolor('xkcd:steel grey')
#     ax.yaxis.pane.set_facecolor('xkcd:steel grey')
#     ax.zaxis.pane.set_facecolor('xkcd:steel grey')

#     # Compute altitudes for the ephemerides
#     alts_all = []
#     for sat in const_ephemerides:
#         ephems = np.array([ephem[1] for ephem in sat])
#         alts_sat = np.linalg.norm(ephems, axis=1) - 6378.137
#         alts_all.extend(alts_sat)

#     # Normalize altitudes for colormap
#     min_alt_all = np.min(alts_all)
#     max_alt_all = np.max(alts_all)
#     norm_alts_all = (alts_all - min_alt_all) / (max_alt_all - min_alt_all)

#     # Setup colorbar
#     print(f"min_alt_all: {min_alt_all}, max_alt_all: {max_alt_all}")
#     cb_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
#     sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=min_alt_all, vmax=max_alt_all))
#     sm._A = []  # Needed for ScalarMappable
#     cbar = fig.colorbar(sm, cax=cb_ax)
#     cbar.set_label('Altitude (km)', fontsize=12, color='black', labelpad=5)
#     cbar.ax.tick_params(labelsize=9, color='black')

#     # Set the title
#     fig.suptitle('Orbital Configuration:' +const +'\n Date:' + str(time.strftime("%d/%m/%y")) + ', ' + str(len(const_ephemerides))+' satellites', fontsize=16, y=0.95, x=0.5, color='black')

#     # Define the animation update function
#     def update(az):
#         ax.cla()
#         ax.view_init(30, az)
#         plot_satellite_data(ax, const_ephemerides, norm_alts_all)
#         return ax

#     print(f"title set to {const}...")
#     # Generate the animation
#     ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), blit=False)
#     print(f"Finished creating animation for {const}...")

#     # Save the animation as a GIF
#     gif_folder = os.path.join('images/constellation_anim/gifs/', const)
#     print(f"Saving gif for {const} at {gif_folder}...")
#     os.makedirs(gif_folder, exist_ok=True)
#     print(f"Saving gif for {const} at {gif_folder}...")
#     ani.save(os.path.join(gif_folder, f'geom_{const}_{time.strftime("%y_%m_%d")}.gif'), writer='imagemagick', fps=10)

#     print(f"Finished creating .gif file for {const}")
    

def generate_geom_gif(const):
    # Process the geometric data for the constellation
    const_ephemerides, _ = process_geom_data(const)
    print(f"Processing geometric data for {const}...")
    print(f"length of const ephemerides: {len(const_ephemerides)}")

    # Set up the figure and axis
    fig = plt.figure(figsize=(6, 6), facecolor='xkcd:steel grey')
    ax = fig.add_subplot(111, projection='3d')

    # Axis setup
    ax.set_box_aspect([1, 1, 1])
    ax.set_zticks(np.arange(-8000, 8001, 2000))
    ax.set_yticks(np.arange(-8000, 8001, 2000))
    ax.set_xticks(np.arange(-8000, 8001, 2000))
    ax.set_xticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax.set_yticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax.set_zticklabels([f'{int(i)}' for i in np.arange(-8000, 8001, 2000)], fontsize=9, color='black')
    ax.set_facecolor('xkcd:steel grey')
    ax.xaxis.pane.set_facecolor('xkcd:steel grey')
    ax.yaxis.pane.set_facecolor('xkcd:steel grey')
    ax.zaxis.pane.set_facecolor('xkcd:steel grey')

    alts_all = [np.linalg.norm(ephem[0][1]) - 6378.137 for ephem in const_ephemerides]

    print(f"first alts_all: {alts_all[:5]}")
    min_alt_all, max_alt_all = np.min(alts_all), np.max(alts_all)

    norm_alts_all = (alts_all - min_alt_all) / (max_alt_all - min_alt_all)

    # Setup colorbar
    cb_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min_alt_all, vmax=max_alt_all))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cb_ax)
    cbar.set_label('Altitude (km)', fontsize=12, color='black', labelpad=5)
    cbar.ax.tick_params(labelsize=9, color='black')

    # Set the title
    fig.suptitle(f'Orbital Configuration: {const}\n Date: {time.strftime("%d/%m/%y")}, {len(const_ephemerides)} satellites', fontsize=16, y=0.95, x=0.5, color='black')

    def update(az):
        ax.cla()
        ax.view_init(30, az)
        plot_satellite_data(ax, const_ephemerides, norm_alts_all)
        return ax

    print(f"title set to {const}...")

    # Save frames and create GIF
    gif_folder = os.path.join('images/constellation_anim/gifs/', const)
    os.makedirs(gif_folder, exist_ok=True)
    frame_files = []
    for az in range(0, 360, 5):
        update(az)
        frame_path = os.path.join(gif_folder, f'frame_{az}.png')
        plt.savefig(frame_path)
        frame_files.append(frame_path)
    plt.close()

    # Compile frames into GIF
    images = [Image.open(frame) for frame in frame_files]
    gif_path = os.path.join(gif_folder, f'geom_{const}_{time.strftime("%y_%m_%d")}.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)

    # Cleanup frame files
    for frame in frame_files:
        os.remove(frame)

    print(f"Finished creating .gif file for {const}")

def process_gtracks_data(const):
    # fetch data and return the data to be plotted
    cwd = os.getcwd()
    animation_folder_path = cwd + '/images/constellation_anim/'
    const_statevec_paths = {}
    imgs_paths = {}
  
    const_statevec_paths = animation_folder_path + "TLEs/" + const + '_const_latest.txt'
    imgs_paths = animation_folder_path + 'ground_tracks/' + const + '/'

    os.makedirs(os.path.dirname(const_statevec_paths), exist_ok=True)
    os.makedirs(os.path.dirname(imgs_paths), exist_ok=True)

    constellation_paths = const_statevec_paths
    constellation_img_paths = imgs_paths

    if not os.path.exists(constellation_paths) or os.path.getmtime(constellation_paths) < (time.time() - 86400):
        print("TLE data is older than 24 hours or not found.")
        print("fetching latest TLE data for:", const)
        pull_constellation_TLEs(tle_txt=constellation_paths, constellation=const)
    else:
        print("TLE data is less than 24 hours old.")
        print("using existing TLE data for:", const)

    # Add a check for whether a plot already exists
    gif_folder = os.path.join('images/constellation_anim/gifs/', const)
    gif_file = os.path.join(gif_folder, f'gtrax_{const}_{time.strftime("%y_%m_%d")}.png')
    if os.path.exists(gif_file):
        print(f"g-tracks plot for {const} already exists. Skipping...")
        return gif_file

    const_ephemerides = []
    individual_TLEs = []
    with open(constellation_paths, 'r') as f:
        for line in f:
            individual_TLEs.append(line.strip())
            if len(individual_TLEs) == 2:
                tle_string = '\n'.join(individual_TLEs)
                TLE_jd = TLE_time(tle_string)
                TLE_jd_plusdt = TLE_jd + 100/1440 # propagate for 100 minutes
                stepsize = 45 # propagate every 20 seconds
                const_ephemerides.append(sgp4_prop_TLE(TLE=tle_string, jd_start=TLE_jd, jd_end=TLE_jd_plusdt, dt=stepsize, alt_series=False))
                individual_TLEs = []
        
    return const_ephemerides, constellation_img_paths

def plot_ground_tracks(const):
    const_ephemerides, constellation_img_paths = process_gtracks_data(const)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidth=0.5) 
    fig.patch.set_facecolor('xkcd:steel grey') 
    ax.spines['geo'].set_visible(False)
    ax.spines['geo'].set_edgecolor('black')
    ax.set_facecolor('xkcd:steel grey')
    ax.gridlines(draw_labels=True, linestyle='--', color='black', alpha=0.5) 

    all_lats = []
    all_lons = []
    all_alts = []
    for sat in const_ephemerides:
        julian_day = sat[0][0] # just get the first MJD
        modified_julian_day = julian_day - 2400000.5
        pos = np.array([ephem[1] for ephem in sat])
        vel = np.array([ephem[2] for ephem in sat])

        # Convert ECI coordinates to ECEF coordinates
        pos, vel = eci2ecef_astropy(pos.T, vel.T, modified_julian_day)
        pos = pos.T
        vel = vel.T

        # Convert ECEF coordinates to lat/lon/alt
        pos_x, pos_y, pos_z = pos.T
        lats, lons, alts = ecef2latlong(pos_x, pos_y, pos_z)

        all_lats.append(lats)
        all_lons.append(lons)
        all_alts.append(alts)

    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)
    alts = np.concatenate(all_alts)

    max_alt = np.max(alts)
    min_alt = np.min(alts)
    norm = mpl.colors.Normalize(vmin=min_alt, vmax=max_alt)
    sm = plt.cm.ScalarMappable(cmap='Spectral', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Altitude (km)', fontsize=10, color='white')
    cbar.ax.tick_params(labelsize=10, color='white')
    # set the colorbar tick labels to white
    cbar.ax.tick_params(axis='x', colors='white')
    cbar.ax.tick_params(axis='y', colors='white')

    #scatter lats and lons and use the color map above to color the points
    ax.scatter(lons, lats, c=alts, cmap='Spectral', s=0.4, transform=ccrs.PlateCarree(), alpha = 0.2)
    #set all plot text to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    num_satellites = len(const_ephemerides)
    ax.set_title(f'Ground tracks: {const} constellation - {time.strftime("%y/%m/%d")}', fontsize=10, color='white') 
    #add a little box to go around the text
    ax.text(0.68, 0.025, f'{num_satellites} active satellites', transform=ax.transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='white', pad=1.0, alpha=0.5))

    os.makedirs(os.path.dirname(constellation_img_paths), exist_ok=True)
    img_file = os.path.join(constellation_img_paths, f'gtrax_{const}_{time.strftime("%y_%m_%d")}.png')
    print("saving image to:", img_file)
    plt.savefig(img_file, dpi=400, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # today = time.strftime("%A")

    # if today == 'Monday':
    #     constellation = 'swarm'
    # elif today == 'Tuesday':
    #     constellation = 'starlink'
    # elif today == 'Wednesday':
    #     constellation = 'planet'
    # elif today == 'Thursday':
    #     constellation = 'spire'
    # elif today == 'Friday':
    #     constellation = 'iridium'
    # elif today == 'Saturday':
    #     constellation = 'oneweb'
    # elif today == 'Sunday':
    #     constellation = 'spire'

    # generate_state_gif('swarm')
    # constellations = ['oneweb', 'starlink', 'planet', 'swarm', 'spire', 'iridium']
    constellations = ['oneweb', 'starlink']
    for constell in constellations:
        generate_geom_gif(constell)