import numpy as np
import pandas as pd

def dist_and_bearing(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    # haversine
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_traveled_km = 6371 * c

    # bearing
    x = np.sin(delta_lon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - \
        np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)

    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360

    return distance_traveled_km, bearing

def bearing_to_cardinal(degrees):
    if pd.isna(degrees):
        return np.nan  # or return 'Unknown' if you want to fill with a label
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((degrees + 22.5) // 45) % 8
    return directions[idx]

def bearing_to_direction(degrees):
    if degrees < 0 or degrees > 360:
        return "Unknown"

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    idx = int((degrees + 22.5) // 45)
    return directions[idx]

def parse_damage(val):
    if isinstance(val, str):
        val = val.strip().upper()
        if val.endswith('K'):
            return float(val[:-1]) * 1_000
        elif val.endswith('M'):
            return float(val[:-1]) * 1_000_000
        elif val.endswith('B'):
            return float(val[:-1]) * 1_000_000_000
        elif val == '0.00':
            return 0.0
        else:
            return np.nan
    return val

def load_data(filename = "list_of_storm_details.csv"):
    df = pd.read_csv(filename)
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'], format = '%d-%b-%y %H:%M:%S')
    df['BEGIN_MONTH'] = df['BEGIN_DATE_TIME'].dt.month
    df['BEGIN_DAY'] = df['BEGIN_DATE_TIME'].dt.day
    df['BEGIN_HOUR'] = df['BEGIN_DATE_TIME'].dt.hour

    df['END_DATE_TIME'] = pd.to_datetime(df['END_DATE_TIME'], format = '%d-%b-%y %H:%M:%S')
    df['END_MONTH'] = df['END_DATE_TIME'].dt.month
    df['END_DAY'] = df['END_DATE_TIME'].dt.day
    df['END_HOUR'] = df['END_DATE_TIME'].dt.hour

    df = df.drop(columns = ['BEGIN_YEARMONTH', 'BEGIN_TIME', 'END_YEARMONTH', 'END_TIME'])

    df['DURATION'] = df['END_DATE_TIME'] - df['BEGIN_DATE_TIME']
    df['DURATION_MINUTES'] = df['DURATION'].dt.total_seconds() / 60
    df = df.drop(columns=['DURATION'])
    df[['DISTANCE_TRAVELED_KM', 'BEARING_DEGREES']] = df.apply(
        lambda row: pd.Series(dist_and_bearing(
            row['BEGIN_LAT'], row['BEGIN_LON'],
            row['END_LAT'], row['END_LON']
        )),
        axis = 1
    )
    df['DIRECTION'] = df['BEARING_DEGREES'].apply(bearing_to_cardinal)
    df['DAMAGE_PROPERTY'] = df['DAMAGE_PROPERTY'].apply(parse_damage)
    df['DAMAGE_CROPS'] = df['DAMAGE_CROPS'].apply(parse_damage)

    '''
    relevant cols in this df:
    0 BEGIN_DAY
    1 END_DAY
    2 EPISODE_ID
    3 EVENT_ID
    4 STATE
    8 EVENT_TYPE
    11 CZ_NAME
    13 BEGIN_DATE_TIME (converted to datetime)
    15 END_DATE_TIME (converted to datetime)
    16 INJURIES_DIRECT
    17 INJURIES_INDIRECT
    18 DEATHS_DIRECT
    19 DEATHS_INDIRECT
    20 DAMAGE_PROPERTY
    21 DAMAGE_CROPS
    23 MAGNITUDE
    40 BEGIN_LAT
    41 BEGIN_LON
    42 END_LAT
    43 END_LON
    44 EPISODE_NARRATIVE
    45 EVENT_NARRATIVE
    47 BEGIN_MONTH (converted, int)
    48 BEGIN_HOUR (converted, int)
    49 END_MONTH (converted, int)
    50 END_HOUR (converted, int)
    51 DURATION_MINUTES (custom, float)
    52 DISTANCE_TRAVELED_KM (custom, float)
    53 BEARING_DEGREES (custom, float)
    54 DIRECTION (custom, object)
    '''
    return df

def main():
    wthr_data = load_data()
    print(wthr_data.info())

if __name__ == "__main__":
    main()