# Hypothetical Storm Damage Predictor

## Overview

The Hypothetical Storm Damage Predictor is a data science and machine learning project that estimates potential storm damages based on the date, time, and geographic location inputs.  
It leverages classification and regression models to predict storm type, expected injuries, deaths, property damage, storm path, and storm duration.

The project is designed to simulate storm event forecasting and assess potential impacts, drawing from historical NOAA storm event data.

---

## Features

- Predicts likely storm event type (e.g., tornado, hail, thunderstorm)
- Estimates direct and indirect injuries
- Forecasts direct and indirect deaths
- Predicts property and crop damage
- Calculates storm magnitude and distance traveled
- Determines storm bearing (direction) and duration
- Small sample dataset included for testing

---

## Technologies Used

- Python
- pandas
- scikit-learn
- geopy
- requests

---

## Dataset

The dataset used for this project is sourced from NOAA's Storm Events Database (public domain):  
[NOAA Storm Events Database](https://www.ncei.noaa.gov/maps/storm-events/)

Due to its large size (~52MB), the full dataset is **not** included in this repository.  
A small sample dataset (list_of_storm_details_sample.csv) is included for testing purposes.

---

## Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/storm-damage-predictor
cd storm-damage-predictor```
Install required libraries:
```pip install -r requirements.txt```
Run the project:
```python user_interface.py```

---

## Future Improvements

- Expand models to include additional storm event types (e.g., hurricanes, winter storms)
- Integrate real-time weather API data for dynamic predictions
- Develop geospatial visualizations of storm paths and damage zones
- Implement a basic alert/warning system based on predicted storm severity
- Optimize machine learning models for improved accuracy and faster predictions

---

## License

This project is licensed under the MIT License.