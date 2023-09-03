import numpy as np
import warnings
import datetime
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.frames import Planes
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential

def sphere_coords(r, granularity):
    """
    Returns the x, y, z coordinates of a sphere with radius r, the granularity is the total number of points in the sphere
    """
    # create a list of angles from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, granularity)
    # create a list of angles from 0 to pi
    phi = np.linspace(0, np.pi, granularity)
    # create a meshgrid of thetas and phis
    theta, phi = np.meshgrid(theta, phi)
    # calculate the x, y, z coordinates of the sphere
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def car2kep(xpos, ypos, zpos, uvel, vvel, wvel, deg=False, arg_l=False):
    """Convert cartesian to keplerian elements.

    Args:
        x (float): x position in km
        y (float): y position in km
        z (float): z position in km
        u (float): x velocity in km/s
        v (float): y velocity in km/s
        w (float): z velocity in km/s
        deg (bool, optional): If True, return angles in degrees. If False, return angles in radians. Defaults to False.
        arg_l (bool, optional): If True, return argument of latitude in degrees. If False, return argument of latitude in radians. Defaults to False.

    Returns:
        tuple: a, e, i, w, W, V, arg_lat
    """
    #TODO: add argument of latitude
    #make the vectors in as astropy Quantity objects
    r = [xpos, ypos, zpos] * u.km
    v = [uvel, vvel, wvel] * u.km / u.s

    #convert to cartesian
    orb = Orbit.from_vectors(Earth, r, v, plane=Planes.EARTH_EQUATOR)

    #convert to keplerian
    if deg == True:

        a = orb.a.value
        e = orb.ecc.value
        i = np.rad2deg(orb.inc.value)
        w = np.rad2deg(orb.raan.value)
        W = np.rad2deg(orb.argp.value)
        V = np.rad2deg(orb.nu.value)
        # arg_lat = np.rad2deg(orb.arglat.value)

        if arg_l == True:
            return a, e, i, w, W, V
        elif arg_l == False:
            return a, e, i, w, W, V

    elif deg == False:
        a = orb.a.value
        e = orb.ecc.value
        i = orb.inc.value
        w = orb.raan.value
        W = orb.argp.value
        V = orb.nu.value
        # arg_lat = orb.arg_lat.value

        if arg_l == True:
            return a, e, i, w, W, V

    return a, e, i, w, W, V                

def kep2car(a, e, i, w, W, V):
    # Suppress the UserWarning for true anomaly wrapping
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        # Create an Orbit object from the Keplerian elements
        orbit = Orbit.from_classical(Earth,
                                     a * u.km,
                                     e * u.one,
                                     i * u.rad,
                                     w * u.rad,
                                     W * u.rad,
                                     V * u.rad,
                                     epoch=Time.now())

    # Get the position and velocity vectors in ECI frame
    pos_vec = orbit.r.value
    vel_vec = orbit.v.value

    # Extract the Cartesian coordinates and velocities
    x, y, z = pos_vec
    vx, vy, vz = vel_vec

    return x, y, z, vx, vy, vz

def tle_convert(tle_dict, display=False):
    """
    Converts a TLE dictionary into the corresponding keplerian elements
    
    Args:
        tle_dict (dict): dictionary of TLE data as provided by the tle_parse function

    Returns:
        keplerian_dict(dict): dictionary containing Keplerian elements
    """

    # Standard gravitational parameter for the Earth
    GM = 398600.4415 * (1e3)**3 # m^3/s^2

    # Convert RAAN from degrees to radians
    RAAN = np.radians(float(tle_dict['right ascension of the ascending node']))
    
    # Convert argument of perigee from degrees to radians
    arg_p = np.radians(float(tle_dict['argument of perigee']))
    
    # Convert mean motion from revolutions per day to radians per second
    mean_motion = float(tle_dict['mean motion']) * (2 * np.pi / 86400)
    
    # Compute the period of the orbit in seconds
    period = 2 * np.pi / mean_motion
    
    # Compute the semi-major axis
    n = mean_motion # mean motion in radians per second
    a = (GM / (n ** 2)) ** (1/3) / 1000 # in km
    
    # Convert mean anomaly from degrees to radians
    M = np.radians(float(tle_dict['mean anomaly']))
    
    # Extract eccentricity as decimal value
    e = float("0." + tle_dict['eccentricity'])
    
    # Convert inclination from degrees to radians
    inclination = np.radians(float(tle_dict['inclination']))
    
    # Initial Guess at Eccentric Anomaly
    if M < np.pi:
        E = M + (e / 2)
    else:
        E = M - (e / 2)

    # Numerical iteration for Eccentric Anomaly
    f = lambda E: E - e * np.sin(E) - M
    fp = lambda E: 1 - e * np.cos(E)
    E = np.float64(E)
    r_tol = 1e-8 # set the convergence tolerance for the iteration
    max_iter = 50 # set the maximum number of iterations allowed
    for it in range(max_iter):
        f_value = f(E)
        fp_value = fp(E)
        E_new = E - f_value / fp_value
        if np.abs(E_new - E) < r_tol:
            E = E_new
            break
        E = E_new
    else:
        raise ValueError("Eccentric anomaly did not converge")
        
    eccentric_anomaly = E

    # Compute True Anomaly
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2),
                                  np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2))

    # Dictionary of Keplerian elements
    keplerian_dict = {'a': a, 'e': e, 'i': inclination, 'RAAN': RAAN, 'arg_p': arg_p, 'true_anomaly': np.degrees(true_anomaly)}
    if display == True:
        print("Keplerian Elements:")
        print("a = {:.2f} km".format(keplerian_dict['a']))
        print("e = {:.2f}".format(keplerian_dict['e']))
        print("i = {:.2f} deg".format(np.degrees(keplerian_dict['i'])))
        print("RAAN = {:.2f} deg".format(np.degrees(keplerian_dict['RAAN'])))
        print("arg_p = {:.2f} deg".format(np.degrees(keplerian_dict['arg_p'])))
        print("true_anomaly = {:.2f} deg".format(np.degrees(keplerian_dict['true_anomaly'])))

    return keplerian_dict

def twoLE_parse(tle_2le):

    """
    Parses a 2LE string (e.g. as provided by Celestrak) and returns all the data in a dictionary.
    Args:
        tle_2le (string): 2LE string to be parsed
    Returns:
        2le_dict (dict): dictionary of all the data contained in the TLE string
    """

    # This function takes a TLE string and returns a dictionary of the TLE data
    tle_lines = tle_2le.split('\n')
    tle_dict = {}
    line_one, line_two = tle_lines[0],tle_lines[1]
    
    #Parse the first line
    tle_dict['line number'] = line_one[0]
    tle_dict['satellite catalog number'] = line_one[2:7]
    tle_dict['classification'] = line_one[7]
    tle_dict['International Designator(launch year)'] = line_one[9:11] 
    tle_dict['International Designator (launch num)'] = line_one[11:14]
    tle_dict['International Designator (piece of launch)'] = line_one[14:17]
    tle_dict['epoch year'] = line_one[18:20]
    tle_dict['epoch day'] = line_one[20:32]
    tle_dict['first time derivative of mean motion(ballisitc coefficient)'] = line_one[33:43]
    tle_dict['second time derivative of mean motion(delta-dot)'] = line_one[44:52]
    tle_dict['bstar drag term'] = line_one[53:61]
    tle_dict['ephemeris type'] = line_one[62]
    tle_dict['element number'] = line_one[63:68]
    tle_dict['checksum'] = line_one[68:69]

    #Parse the second line (ignore the line number, satellite catalog number, and checksum)
    tle_dict['inclination'] = line_two[8:16]
    tle_dict['right ascension of the ascending node'] = line_two[17:25]
    tle_dict['eccentricity'] = line_two[26:33]
    tle_dict['argument of perigee'] = line_two[34:42]
    tle_dict['mean anomaly'] = line_two[43:51]
    tle_dict['mean motion'] = line_two[52:63]
    tle_dict['revolution number at epoch'] = line_two[63:68]

    return tle_dict

def TLE_time(TLE):
    """Find the time of a TLE in julian day format"""
    #find the epoch section of the TLE
    epoch = TLE[18:32]
    #convert the first two digits of the epoch to the year
    year = 2000+int(epoch[0:2])
    
    # the rest of the digits are the day of the year and fractional portion of the day
    day = float(epoch[2:])
    #convert the day of the year to a day, month, year format
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
    #convert the date to a julian date
    jd = (date - datetime.datetime(1858, 11, 17)).total_seconds() / 86400.0 + 2400000.5
    return jd

def utc_to_jd(time_stamps):
    """Converts UTC time to Julian Date.
    Args: 
        time_stamps(list): list of UTC times datetime.datetime objects that need to be converted to JD.
    Returns:
        jd(list): is a list of Julian Date times.
    """
    UTC_string = []
    for i in range(0,len(time_stamps),1):
        UTC_string.append(time_stamps[i].strftime('%Y-%m-%d %H:%M:%S'))

    t = Time(UTC_string, format='iso', scale='utc') #astropy time object
    jd = t.to_value('jd', 'long') #convert to jd

    jd_vals = []
    for i in range (0, len(jd),1):
        jd_vals.append(float(jd[i]))
    
    return jd_vals

def ecef2latlong(P_x, P_y, P_z):
    """
    Convert between Earth-Centred-Earth-Fixed cartesian coordinates to lat-long-height
    Assumes a spherical planet with constant radius of 6378.137km.

    Args:
    P_x (float) = x-coordinate of the point p
    P_y (float) = y-coordinate of the point p
    P_z (float) = z-coordinate of the point p

    Returns:
    lat (float) = latitude of the point p
    long (float) = longitude of the point p
    h (float) = height of the point p from the surface of the earth
    """

    Re = 6378.137  # Earth radius in kilometers

    # Calculating Longitude
    Lambda_rad = np.arctan2(P_y, P_x)  # This is in radians
    long = np.degrees(Lambda_rad)  # Converting to degrees

    # Calculating Latitude
    Phi_rad = np.arctan2(P_z, np.sqrt(P_x**2 + P_y**2))  # This is in radians
    lat = np.degrees(Phi_rad)  # Converting to degrees

    # Calculating the radius vector magnitude
    r_mag = np.sqrt((P_x**2) + (P_y**2) + (P_z**2))

    h = r_mag - Re  # Calculating the height
    return lat, long, h

def eci2ecef_astropy(eci_pos, eci_vel, mjd):

    # Convert MJD to isot format for Astropy
    time_utc = Time(mjd, format="mjd", scale='utc')

    # Convert ECI position and velocity to ECEF coordinates using Astropy
    eci_cartesian = CartesianRepresentation(eci_pos.T * u.km)
    eci_velocity = CartesianDifferential(eci_vel.T * u.km / u.s)
    gcrs_coords = GCRS(eci_cartesian.with_differentials(eci_velocity), obstime=time_utc)
    itrs_coords = gcrs_coords.transform_to(ITRS(obstime=time_utc))

    # Get ECEF position and velocity from Astropy coordinates
    ecef_pos = np.column_stack((itrs_coords.x.value, itrs_coords.y.value, itrs_coords.z.value))
    ecef_vel = np.column_stack((itrs_coords.v_x.value, itrs_coords.v_y.value, itrs_coords.v_z.value))

    return ecef_pos, ecef_vel
