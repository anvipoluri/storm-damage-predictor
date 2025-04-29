from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import requests
import time
from data_cleaning import load_data
from knn_define import knn_mod
from lin_re_define import linre_mod
import pandas as pd
import requests
from geopy.geocoders import Nominatim
import random
import joblib
import calendar

def bearing_to_direction(degrees):
    if degrees < 0 or degrees > 360:
        return "Unknown"
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    idx = int((degrees + 22.5) // 45)
    return directions[idx]

def address_trim(address):
    parts = [part.strip() for part in address.split(",")]
    if len(parts) < 2:
        return parts[0]

    town_or_place = parts[0].title()

    state_guess = ""
    for part in parts[1:]:
        lower = part.lower()
        if "county" in lower:
            continue
        if "united states" in lower:
            continue
        if any(char.isdigit() for char in part):
            continue
        state_guess = part.strip()
        break

    if state_guess:
        return f"{town_or_place}, {state_guess}"
    else:
        return f"{town_or_place}"

def get_coordinates(place_name, max_retries = 3):
    geolocator = Nominatim(user_agent="storm_predictor")
    clean_input = place_name.strip()

    if clean_input.replace(",", "").replace(" ", "").isdigit():
        print("Invalid location input (only numbers). Please enter a city or place name\n")
        return None, None

    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(clean_input, timeout = 5)
            if location:
                address = location.address.lower()
                short_address = address_trim(address)
                return short_address, location.latitude, location.longitude
            else:
                print("Can't find that location.. Please try again\n")
                return None, None

        except (GeocoderUnavailable, GeocoderTimedOut, requests.exceptions.ConnectionError):
            print(f"Geocoding service timeout (attempt {attempt + 1}/{max_retries}. Retrying...\n")
            time.sleep(random.uniform(1, 2))
            break

    print(
        "Geocoding service unavailable after multiple attempts. Please check your connection or try "
        "a different place.\n")
    return None, None

def predict_event_type(knn_model, month, day, hour, lat, lon):
    input_df = pd.DataFrame([{
        'BEGIN_MONTH': int(month),
        'BEGIN_DAY': int(day),
        'BEGIN_HOUR': int(hour),
        'BEGIN_LAT': float(lat),
        'BEGIN_LON': float(lon)
    }])
    return knn_model.predict(input_df)[0]

def valid_mo(month_input):
    month_input = month_input.lower().strip()
    month_map = {name.lower(): num for num, name in enumerate(calendar.month_name) if name}
    abbr_map = {abbr.lower(): num for num, abbr in enumerate(calendar.month_abbr) if abbr}

    if month_input in month_map:
        return month_map[month_input]
    if month_input in abbr_map:
        return abbr_map[month_input]
    if month_input.isdigit():
        month_num = int(month_input)
        if 1 <= month_num <= 12:
            return month_num
    raise ValueError("Invalid month format")

def valid_day(month_num, day_n):
    _, max_day = calendar.monthrange(2025, month_num)
    return 1 <= day_n <= max_day

def valid_hr(hour_n):
    return 0<= hour_n <= 23

def user_input():
    while True:
        while True:
            # month
            try:
                raw_mo = input("Enter month (e.g. apr): ")
                month = valid_mo(raw_mo)
                break
            except ValueError as ve:
                print(f"Input error: {ve}\nPlease try again.\n")

        while True:
            try:
                day = int(input("Enter day (e.g. 12): "))
                if not valid_day(month, day):
                    raise ValueError("Invalid day for the given month")
                break
            except ValueError as ve:
                print(f"Input error: {ve}\nPlease try again.\n")

        while True:
            try:
                hour = int(input("Enter hour (24h, e.g. 14): "))
                if not valid_hr(hour):
                    raise ValueError("Invalid hour")
                break
            except ValueError as ve:
                print(f"Input error: {ve}\nPlease try again.\n")

        while True:
            place = input("Enter storm location (city, county, or place name): ")
            town_name, lat, lon = get_coordinates(place)
            if lat is not None and lon is not None:
                break

        print(f"\nAll inputs received!")
        return month, day, hour, lat, lon, town_name

def full_ui(knn_mod, lr_mod, lr_event_encoder):
    print("Hypothetical Storm Damages calculator")
    month, day, hour, lat, lon, town_name = user_input()

    print(f"Calculating storm damages for {town_name.title()}.")

    event_type = predict_event_type(knn_mod, month, day, hour, lat, lon)

    event_encoded = lr_event_encoder.transform([event_type])[0]

    user_X = pd.DataFrame([{
        'PREDICTED_EVENT_TYPE': event_encoded,
        'BEGIN_MONTH': int(month),
        'BEGIN_DAY': int(day),
        'BEGIN_HOUR': int(hour),
        'BEGIN_LAT': float(lat),
        'BEGIN_LON': float(lon)
    }])

    prediction = lr_mod.predict(user_X)[0].tolist()

    prediction = {
        'INJURIES_DIRECT': max(prediction[0], 0),
        'INJURIES_INDIRECT': max(prediction[1], 0),
        'DEATHS_DIRECT': max(prediction[2], 0),
        'DEATHS_INDIRECT': max(prediction[3], 0),
        'DAMAGE_PROPERTY': max(prediction[4], 0),
        'DAMAGE_CROPS': max(prediction[5], 0),
        'MAGNITUDE': max(prediction[6],0),
        'END_LAT': prediction[7],
        'END_LON': prediction[8],
        'DURATION_MINUTES': max(prediction[9], 0),
        'DISTANCE_TRAVELED_KM': max(prediction[10], 0),
        'BEARING_DEGREES': prediction[11],
        'DIRECTION': prediction[12]
    }

    print(f"\nPredicted storm outcomes:")
    print(f"PREDICTED EVENT TYPE: {event_type}")
    for key, val in prediction.items():
        if key == 'DURATION_MINUTES':
            if val < 0.5:
                print("DURATION_MINUTES: Instantaneous (less than 1 minute storm)")
            else:
                print(f"DURATION_MINUTES: {val:.2f}")
        elif key == 'BEARING_DEGREES':
            direction = bearing_to_direction(val)
            print(f"BEARING_DEGREES: {val:.2f} {direction}")
        else:
            print(f"{key}: {val:.2f}")

def main():
    knn = joblib.load("knn_model.pkl")
    lr_mod = joblib.load("linre_model.pkl")
    lr_event_encoder = joblib.load("linre_encoder.pkl")

    full_ui(knn, lr_mod, lr_event_encoder)

if __name__ == "__main__":
    main()