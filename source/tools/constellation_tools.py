import os
import configparser
import requests
import datetime

from source.tools.conversions import kep2car, tle_convert, utc_to_jd

class MyError(Exception):
    def __init___(self, args):
        Exception.__init__(
            self, "my exception was raised with arguments {0}".format(args)
        )
        self.args = args

def SpaceTrack_authenticate():
    """Authenticate with SpaceTrack using the credentials stored in the config file. 

    Returns:
        str: SpaceTrack username
        str: SpaceTrack password

    """
    password = os.environ.get("SLTRACK_PWD")
    username = os.environ.get("SLTRACK_USR")
    if not all((username, password)):

        config = configparser.ConfigParser()
        config.read("/Users/charlesc/Documents/GitHub/Astrodynamics/SLTrack.ini")
        username = config.get("configuration", "username")
        password = config.get("configuration", "password")

    assert ( username is not None and password is not None), "Please specify a username and password for SpaceTrack"
    
    return username, password

def pull_constellation_statevecs(cart_txt, constellation):
    """Populate a text file with the cartesian states of all the latest available satellites of the specified constellation using the SpaceTrack API. Largely based on the code from the SpaceTrack API documentation,
     Tutorial by Andrew Stokes (2019). Data is from TLEs.

    Args:
        cart_txt (regexp): file path of .txt file to write the states to
        constellation (str, optional): constellation to pull. Defaults to 'oneweb'.

    """
    # See https://www.space-track.org/documentation for details on REST queries

    # have to get these from spacetrack.org
    constellation_cat_names = {"starlink": "STARLINK", "oneweb": "ONEWEB", "planet": "FLOCK", "swarm": "SpaceBEE", "spire": "LEMUR", "iridium": "IRIDIUM"}

    available_constellations = list(constellation_cat_names.keys())
    if constellation not in available_constellations:
        raise ValueError("Invalid constellation name. Select one of: %s" % available_constellations)

    assert (
        cart_txt is not None
    ), "Please specify a text file to write the cartesian states to"
    assert cart_txt.endswith(
        ".txt"
    ), "Please specify a text file to write the cartesian states to"
    assert type(cart_txt) == str, "path must be a string"

    uriBase = "https://www.space-track.org"
    requestLogin = "/ajaxauth/login"
    requestCmdAction = "/basicspacedata/query"
    if constellation in available_constellations:
        cat_name = constellation_cat_names[constellation]
        requestFindOWs = f"/class/tle_latest/NORAD_CAT_ID/>40000/ORDINAL/1/OBJECT_NAME/{cat_name}~~/format/tle/orderby/NORAD_CAT_ID%20asc"

    # Find credentials for the SpaceTrack API
    
    username, password = SpaceTrack_authenticate()
    siteCred = {'identity': username, 'password': password}
    with open(cart_txt, "w") as f:
        # use requests package to drive the RESTful
        # session with space-track.org

        with requests.Session() as session:
            # run the session in a with block to
            # force session to close if we exit

            # need to log in first. note that we get a 200
            # to say the web site got the data, not that we are logged in
            resp = session.post(uriBase + requestLogin, data=siteCred)
            if resp.status_code != 200:
                raise MyError(resp, "POST fail on login")

            # this query picks up all OneWeb satellites from the catalog.
            # Note - a 401 failure shows you have bad credentials
            resp = session.get(uriBase + requestCmdAction + requestFindOWs)
            if resp.status_code != 200:
                raise MyError(
                    resp, "GET fail on request for satellites"
                )

            output = resp.text

        # split the output into lines
        all_lines = output.splitlines()
        # put every two lines into a list
        line_pairs = [
            all_lines[i:i + 2] for i in range(0, len(all_lines), 2)
        ]

        for tle in range(0, len(line_pairs), 1):
            line_one, line_two = line_pairs[tle][0], line_pairs[tle][1]

            tle_dict = {}

            # Parse the first line
            tle_dict["line number"] = line_one[0]
            tle_dict["satellite catalog number"] = line_one[2:7]
            tle_dict["classification"] = line_one[7]
            tle_dict["International Designator(launch year)"] = line_one[9:11]
            tle_dict["International Designator (launch num)"] = line_one[11:14]
            tle_dict["International Designator (piece of launch)"] = line_one[
                14:17
            ]
            tle_dict["epoch year"] = line_one[18:20]
            tle_dict["epoch day"] = line_one[20:32]
            tle_dict[
                "first time derivative of mean motion(ballisitc coefficient)"
            ] = line_one[33:43]
            tle_dict[
                "second time derivative of mean motion(delta-dot)"
            ] = line_one[44:52]
            tle_dict["bstar drag term"] = line_one[53:61]
            tle_dict["ephemeris type"] = line_one[62]
            tle_dict["element number"] = line_one[63:68]
            tle_dict["checksum"] = line_one[68:69]

            # Parse the second line (ignore the line number,
            # satellite catalog number, and checksum)
            tle_dict["inclination"] = line_two[8:16]
            tle_dict["right ascension of the ascending node"] = line_two[17:25]
            tle_dict["eccentricity"] = line_two[26:33]
            tle_dict["argument of perigee"] = line_two[34:42]
            tle_dict["mean anomaly"] = line_two[43:51]
            tle_dict["mean motion"] = line_two[52:63]
            tle_dict["revolution number at epoch"] = line_two[63:68]

            kep_elems = tle_convert(tle_dict)
            print("kepelems:",kep_elems)

            x_car, y_car, z_car, u_car, v_car, w_car = kep2car(
                a=kep_elems["a"],
                e=kep_elems["e"],
                i=kep_elems["i"],
                w=kep_elems["RAAN"],
                W=kep_elems["arg_p"],
                V=kep_elems["true_anomaly"],
            )
            state_i = [x_car, y_car, z_car, u_car, v_car, w_car]
            utc_stamp = datetime.datetime.now()
            jd_stamp = utc_to_jd([utc_stamp])
            f.write(
                "sat:"
                + str(tle_dict["satellite catalog number"])
                + "ECI:"
                + str(state_i)
                + "jd_time:"
                + str(jd_stamp)
                + "\n"
            )
    f.close()

def pull_constellation_TLEs(tle_txt, constellation):
    """Populate a text file with the TLEs of all the latest available satellites of the specified constellation using the SpaceTrack API. Largely based on the code from the SpaceTrack API documentation,
     Tutorial by Andrew Stokes (2019).

    Args:
        tle_txt (regexp): file path of .txt file to write the TLEs to
        constellation (str, optional): constellation to pull. Defaults to 'oneweb'.

    """
    # See https://www.space-track.org/documentation for details on REST queries

    # have to get these from spacetrack.org
    constellation_cat_names = {"starlink": "STARLINK", "oneweb": "ONEWEB", "planet": "FLOCK", "swarm": "SpaceBEE", "spire": "LEMUR", "iridium": "IRIDIUM"}

    available_constellations = list(constellation_cat_names.keys())
    if constellation not in available_constellations:
        raise ValueError("Invalid constellation name. Select one of: %s" % available_constellations)

    assert (
        tle_txt is not None
    ), "Please specify a text file to write the TLEs to"
    assert tle_txt.endswith(
        ".txt"
    ), "Please specify a text file to write the TLEs to"
    assert type(tle_txt) == str, "path must be a string"

    uriBase = "https://www.space-track.org"
    requestLogin = "/ajaxauth/login"
    requestCmdAction = "/basicspacedata/query"
    if constellation in available_constellations:
        cat_name = constellation_cat_names[constellation]
        requestFindConstellation = f"/class/tle_latest/NORAD_CAT_ID/>40000/ORDINAL/1/OBJECT_NAME/{cat_name}~~/format/tle/orderby/NORAD_CAT_ID%20asc"

    # Find credentials for the SpaceTrack API
    username, password = SpaceTrack_authenticate()
    siteCred = {'identity': username, 'password': password}
    with open(tle_txt, "w") as f:
        # use requests package to drive the RESTful
        # session with space-track.org

        with requests.Session() as session:
            # run the session in a with block to
            # force session to close if we exit

            # need to log in first. note that we get a 200
            # to say the web site got the data, not that we are logged in
            resp = session.post(uriBase + requestLogin, data=siteCred)
            if resp.status_code != 200:
                raise MyError(resp, "POST fail on login")

            # this query picks up all satellites from the specified constellation from the catalog.
            # Note - a 401 failure shows you have bad credentials
            resp = session.get(uriBase + requestCmdAction + requestFindConstellation)
            if resp.status_code != 200:
                raise MyError(
                    resp, "GET fail on request for satellites"
                )

            output = resp.text

        # split the output into lines
        all_lines = output.splitlines()
        # put every two lines into a list
        line_pairs = [
            all_lines[i:i + 2] for i in range(0, len(all_lines), 2)
        ]

        for tle_pair in line_pairs:
            line_one, line_two = tle_pair[0], tle_pair[1]

            # Write TLE pair to file
            f.write(line_one + "\n" + line_two + "\n")

    f.close()