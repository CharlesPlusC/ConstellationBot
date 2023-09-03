import os
import json
import numpy as np
from datetime import datetime
from source.tools.constellation_tools import pull_constellation_statevecs, pull_constellation_TLEs
from source.tools.conversions import car2kep, kep2car, tle_convert, twoLE_parse, TLE_time

def fetch_sat_info(const, anim, format):
    """
    Fetch satellite information for the given constellation and format (TLE or state vector).

    :param const: The constellation name.
    :type const: str
    :param format: The format of the satellite information ("TLE" or "statevecs").
    :type format: str
    :return: A tuple containing the path to the constellation data and the path to the constellation image.
    :rtype: tuple
    """
    possible_fmts = ["TLE", "statevecs"]
    if format not in possible_fmts:
        raise ValueError("format must be one of: " + str(possible_fmts))
    
    possbile_anims = ["latest_state", "current_geometry", "ground_tracks"]
    if anim not in possbile_anims:
        raise ValueError("anim must be one of: " + str(possbile_anims))
    # fetch data and return the data to be plotted
    cwd = os.getcwd()
    animation_folder_path = cwd + '/images/constellation_anim/'
    const_statevec_paths = {}
    imgs_paths = {}
    
    if format == "TLE":
        const_statevec_paths = animation_folder_path + "TLEs/" + const + '_const_latest.txt'
    elif format == "statevecs":
        const_statevec_paths = animation_folder_path + "state_vecs/" + const + '_const_latest.txt'
    
    imgs_paths = animation_folder_path + anim + '/'+ const + '/'

    os.makedirs(os.path.dirname(const_statevec_paths), exist_ok=True)
    os.makedirs(os.path.dirname(imgs_paths), exist_ok=True)

    constellation_paths = const_statevec_paths
    constellation_img_paths = imgs_paths

    #removed the ifloop that checks if the file exists or is older than 1 day
    if format == "TLE":
        print("fetching latest TLEs for:", const)
        pull_constellation_TLEs(tle_txt=constellation_paths, constellation=const)
        print("calculating constellation statistics", const)
        total_satellites, altitudes, inclinations = calculate_TLE_stats(constellation_paths)
        altitude_counts, inclination_counts = classify_satellites(altitudes, inclinations)
        print("populating json file for:", const)
        populate_json(total_satellites = total_satellites, altitude_counts = altitude_counts, inclination_counts = inclination_counts,constellation_name=const)
    elif format == "statevecs":
        print("fetching latest state vectors for:", const)
        pull_constellation_statevecs(cart_txt=constellation_paths, constellation=const)
        print("calculating constellation statistics", const)
        total_satellites, altitudes, inclinations = calculate_statevec_stats(constellation_paths)
        altitude_counts, inclination_counts = classify_satellites(altitudes, inclinations)
        print("populating json file for:", const)
        populate_json(total_satellites = total_satellites, altitude_counts = altitude_counts, inclination_counts = inclination_counts,constellation_name=const)


    return constellation_paths, constellation_img_paths

def pos_vel_from_statevecs(path):
    """
    Extract position and velocity from the state vectors at the given path.

    :param path: The path to the state vectors file.
    :type path: str
    :return: A tuple containing the ECI positions, ECI velocities, and altitudes.
    :rtype: tuple
    """
    # read the file containing the state vectors and return the position and velocity
    with open(path, 'r') as f:
        lines = f.readlines()
        #split everytime there is a colon
        lines = [line.split(':') for line in lines]
        # the eci position is the 3rd element in the list
        eci_coords = [line[2] for line in lines]
        # remove the last 7 characters of the string
        eci_coords = [pos[:-8] for pos in eci_coords]
        # remove the first 2 characters of the string
        eci_coords = [pos[1:] for pos in eci_coords]
        # split them by element
        eci_coords = [pos.split(',') for pos in eci_coords]
        #convert to floats
        eci_coords = [[float(pos) for pos in pos] for pos in eci_coords]
        # keep only the first 3 elements of each row to get the position
        eci_pos = [pos[:3] for pos in eci_coords]
        # keep only the last 3 elements of each row to get the velocity
        eci_vel = [pos[3:] for pos in eci_coords]
        # take the np.linalg.norm of each row to get the magnitude and remove 6378.137Km to get the altitude
        alts = [np.linalg.norm(pos) - 6378.137 for pos in eci_pos]
        f.close()
    return eci_pos, eci_vel, alts

def pos_vel_from_TLEs(path):
    """
    Extract position and velocity from the TLEs at the given path.

    :param path: The path to the TLEs file.
    :type path: str
    :return: A tuple containing the ECI positions, ECI velocities, and altitudes.
    :rtype: tuple
    """
    eci_pos = []
    eci_vel = []
    alts = []
    individual_TLEs = [] # list to hold the two lines of the TLE
    with open(path, 'r') as f:
        for line in f:
            individual_TLEs.append(line.strip())
            if len(individual_TLEs) == 2:
                tle_string = '\n'.join(individual_TLEs)
                TLE_jd = TLE_time(tle_string)
                tle_dict = twoLE_parse(tle_string)
                kepler_dict = tle_convert(tle_dict)
                x,y,z,u,v,w = kep2car(kepler_dict['a'], kepler_dict['e'], kepler_dict['i'], kepler_dict['RAAN'], kepler_dict['arg_p'], kepler_dict['true_anomaly'])
                individual_TLEs = []
                eci_pos.append([x,y,z])
                eci_vel.append([u,v,w])
                alts.append(np.linalg.norm([x,y,z]) - 6378.137)
    f.close()
    return eci_pos, eci_vel, alts

def calculate_statevec_stats(constellation_path):
    """
    Calculate satellite statistics from the state vectors at the given path.

    :param constellation_path: The path to the state vectors file.
    :type constellation_path: str
    :return: A tuple containing the total number of satellites, altitudes, and inclinations.
    :rtype: tuple
    """
    eci_pos, eci_vel, alts = pos_vel_from_statevecs(constellation_path)
    altitudes = alts
    inclinations = []
    for i in range(len(eci_pos)):
        x, y, z = eci_pos[i]
        u, v, w = eci_vel[i]
        a,e,i,w,W,V = car2kep(x, y, z, u, v, w, deg=True, arg_l=False)
        inclinations.append(i)
    total_satellites = len(eci_pos)

    return total_satellites, altitudes, inclinations

def calculate_TLE_stats(constellation_path):
    """
    Calculate satellite statistics from the TLEs at the given path.

    :param constellation_path: The path to the TLEs file.
    :type constellation_path: str
    :return: A tuple containing the total number of satellites, altitudes, and inclinations.
    :rtype: tuple
    """
    eci_pos, eci_vel, alts = pos_vel_from_TLEs(constellation_path)
    altitudes = alts
    inclinations = []
    for i in range(len(eci_pos)):
        x, y, z = eci_pos[i]
        u, v, w = eci_vel[i]
        a,e,i,w,W,V = car2kep(x, y, z, u, v, w, deg=True, arg_l=False)
        inclinations.append(i)
    total_satellites = len(eci_pos)
    
    return total_satellites, altitudes, inclinations

def classify_satellites(altitudes, inclinations):
    """
    Classify satellites based on their altitudes and inclinations.

    :param altitudes: A list of satellite altitudes.
    :type altitudes: list
    :param inclinations: A list of satellite inclinations.
    :type inclinations: list
    :return: A tuple containing dictionaries with altitude and inclination counts.
    :rtype: tuple
    """
    altitude_ranges = [f"{i}-{i+100}" for i in range(100, 2000, 100)]
    inclination_ranges = [f"{i}-{i+10}" for i in range(0, 180, 10)]
    
    altitude_counts = {range_str: 0 for range_str in altitude_ranges}
    inclination_counts = {range_str: 0 for range_str in inclination_ranges}

    for altitude, inclination in zip(altitudes, inclinations):
        for alt_range in altitude_ranges:
            low, high = map(int, alt_range.split('-'))
            if low <= altitude < high:
                altitude_counts[alt_range] += 1

        for inc_range in inclination_ranges:
            low, high = map(int, inc_range.split('-'))
            if low <= inclination < high:
                inclination_counts[inc_range] += 1

    return altitude_counts, inclination_counts

def populate_json(total_satellites, altitude_counts, inclination_counts,constellation_name,json_file="source/twitterbot/info_JSON/constellation_history.json",):
    """
    Populate a JSON file with the provided satellite information.

    :param total_satellites: The total number of satellites in the constellation.
    :type total_satellites: int
    :param altitude_counts: A dictionary containing altitude count ranges.
    :type altitude_counts: dict
    :param inclination_counts: A dictionary containing inclination count ranges.
    :type inclination_counts: dict
    :param constellation_name: The name of the constellation.
    :type constellation_name: str
    :param json_file: The path to the JSON file (default is "source/twitterbot/info_JSON/constellation_history.json").
    :type json_file: str, optional
    """
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        try:
            with open(json_file, "r") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            data = {"constellations": {}}
    else:
        data = {"constellations": {}}

    if constellation_name not in data["constellations"]:
        data["constellations"][constellation_name] = {"history": []}

    current_data = {
        "timestamp": datetime.now().isoformat(),
        "total_satellites": total_satellites,
        "altitude_counts": altitude_counts,
        "inclination_counts": inclination_counts
    }

    data["constellations"][constellation_name]["history"].append(current_data)

    with open(json_file, "w") as file:
        json.dump(data, file, indent=2)


def load_json_data(json_file="source/twitterbot/info_JSON/constellation_history.json"):
    """
    Load JSON data from the specified file.
    
    Args:
        json_file (str): The path of the JSON file. Defaults to "source/twitterbot/info_JSON/constellation_history.json".

    Returns:
        dict: Loaded JSON data.
    """

    with open(json_file, "r") as file:
        data = json.load(file)
    return data

def get_previous_records(constellation_name, data):
    """
    Get the last and second last records for a given constellation from the loaded data.
    
    Args:
        constellation_name (str): The name of the constellation.
        data (dict): Loaded JSON data.

    Returns:
        tuple: A tuple containing the last and second last records or (None, None) if not found.
    """

    if constellation_name in data["constellations"]:
        history = data["constellations"][constellation_name]["history"]
        if len(history) >= 2:
            last_record = history[-1]
            second_last_record = history[-2]
            return last_record, second_last_record
    return None, None

def calculate_altitude_diff(last_record, second_last_record):
    """
    Calculate the altitude difference between the last and second last records.

    Args:
        last_record (dict): The last record.
        second_last_record (dict): The second last record.

    Returns:
        dict: A dictionary containing the differences for each altitude range or None if any of the input records is missing.
    """
    if not last_record or not second_last_record:
        return None
    last_alt_counts = last_record["altitude_counts"]
    second_last_alt_counts = second_last_record["altitude_counts"]
    diff_alt_counts = {
        key: last_alt_counts[key] - second_last_alt_counts[key] for key in last_alt_counts
    }
    return diff_alt_counts

def calculate_inclination_diff(last_record, second_last_record):
    """
    Calculate the inclination difference between the last and second last records.

    Args:
        last_record (dict): The last record.
        second_last_record (dict): The second last record.

    Returns:
        dict: A dictionary containing the differences for each inclination range or None if any of the input records is missing.
    """
    if not last_record or not second_last_record:
        return None
    last_inc_counts = last_record["inclination_counts"]
    second_last_inc_counts = second_last_record["inclination_counts"]
    diff_inc_counts = {
        key: last_inc_counts[key] - second_last_inc_counts[key] for key in last_inc_counts
    }
    return diff_inc_counts

def calculate_total_satellites_diff(last_record, second_last_record):
    """
    Calculate the difference in total number of satellites between the last and second last records.

    Args:
        last_record (dict): The last record.
        second_last_record (dict): The second last record.

    Returns:
        int: The difference in total number of satellites or None if any of the input records is missing.
    """

    if not last_record or not second_last_record:
        return None
    return last_record["total_satellites"] - second_last_record["total_satellites"]

def calculate_date_difference(last_record, second_last_record):
    """
    Calculate the difference in dates between the last and second last records.

    Args:
        last_record (dict): The last record.
        second_last_record (dict): The second last record.

    Returns:
        int: The difference in days between the dates of the two records or None if any of the input records is missing.
    """
    if not last_record or not second_last_record:
        return None
    last_date = datetime.fromisoformat(last_record["timestamp"])
    second_last_date = datetime.fromisoformat(second_last_record["timestamp"])
    return (last_date - second_last_date).days

def constellation_difference(constellation_name, json_file="source/twitterbot/info_JSON/constellation_history.json"):
    """
    Calculate the differences in altitude, inclination, total satellites, and date for a given constellation.

    Args:
        constellation_name (str): The name of the constellation.
        json_file (str): The path of the JSON file. Defaults to "source/twitterbot/info_JSON/constellation_history.json".

    Returns:
        tuple: A tuple containing the differences in altitude counts, inclination counts, total satellites, and date (in days).
    """
    data = load_json_data(json_file)
    last_record, second_last_record = get_previous_records(constellation_name, data)

    alt_diff = calculate_altitude_diff(last_record, second_last_record)
    inc_diff = calculate_inclination_diff(last_record, second_last_record)
    total_sat_diff = calculate_total_satellites_diff(last_record, second_last_record)
    date_diff = calculate_date_difference(last_record, second_last_record)

    return alt_diff, inc_diff, total_sat_diff, date_diff

def constellation_difference_non_zero(constellation_name, json_file="source/twitterbot/info_JSON/constellation_history.json"):
    """
    Calculate non-zero differences in altitude, inclination, total satellites, and date for a given constellation.

    Args:
        constellation_name (str): The name of the constellation.
        json_file (str): The path of the JSON file. Defaults to "source/twitterbot/info_JSON/constellation_history.json".

    Returns:
        tuple: A tuple containing non-zero differences in altitude counts, inclination counts, total satellites, and date (in days). If all differences are zero, return 0.
    """
    data = load_json_data(json_file)
    last_record, second_last_record = get_previous_records(constellation_name, data)

    alt_diff = calculate_altitude_diff(last_record, second_last_record)
    inc_diff = calculate_inclination_diff(last_record, second_last_record)
    total_sat_diff = calculate_total_satellites_diff(last_record, second_last_record)
    date_diff = calculate_date_difference(last_record, second_last_record)
    print("altdiff, incdiff, sat diff, datediff",alt_diff, inc_diff, total_sat_diff, date_diff)
    non_zero_alt_diff = {k: v for k, v in alt_diff.items() if v != 0} if alt_diff else None
    non_zero_inc_diff = {k: v for k, v in inc_diff.items() if v != 0} if inc_diff else None
    print(non_zero_alt_diff, non_zero_inc_diff, total_sat_diff, date_diff)

    if non_zero_alt_diff or non_zero_inc_diff or total_sat_diff or date_diff:
        return non_zero_alt_diff, non_zero_inc_diff, total_sat_diff, date_diff
    else:
        return 0

if __name__ == "__main__":
    pass
    # const = "oneweb"
    # fetch_sat_info(const, format="statevecs", anim = "latest_state"
    # constellation_name = "oneweb"
    # result = constellation_difference_non_zero(constellation_name)
    # if len(result) == 1:
    #     print("No change")
    # else:
    #     print("altitude diff: ", result[0])
    #     print("inclination diff: ", result[1])
    #     print("total satellites diff: ", result[2])
    #     print("date diff: ", result[3])