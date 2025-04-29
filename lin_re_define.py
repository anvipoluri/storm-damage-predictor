from data_cleaning import load_data
from knn_define import knn_mod
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib

def linre_mod(knn_results, df):

    clippers = [
        'INJURIES_DIRECT', 'INJURIES_INDIRECT',
        'DEATHS_DIRECT', 'DEATHS_INDIRECT',
        'DAMAGE_PROPERTY', 'DAMAGE_CROPS',
        'DURATION_MINUTES', 'DISTANCE_TRAVELED_KM'
    ]

    targets = [
        'INJURIES_DIRECT','INJURIES_INDIRECT',
        'DEATHS_DIRECT','DEATHS_INDIRECT',
        'DAMAGE_PROPERTY','DAMAGE_CROPS',
        'MAGNITUDE',
        'END_LAT','END_LON',
        'DURATION_MINUTES',
        'DISTANCE_TRAVELED_KM','BEARING_DEGREES','DIRECTION'
    ]

    features = [
        'PREDICTED_EVENT_TYPE',
        'BEGIN_MONTH', 'BEGIN_DAY', 'BEGIN_HOUR',
        'BEGIN_LAT', 'BEGIN_LON'
    ]

    merged = pd.merge(
        knn_results,
        df[targets + ['BEGIN_MONTH', 'BEGIN_DAY', 'BEGIN_HOUR', 'BEGIN_LAT', 'BEGIN_LON']],
        how='inner',
        on=['BEGIN_MONTH', 'BEGIN_DAY', 'BEGIN_HOUR', 'BEGIN_LAT', 'BEGIN_LON']
    ).copy()

    merged = merged.dropna(subset=targets).copy()

    merged['DIRECTION'] = LabelEncoder().fit_transform(merged['DIRECTION'].astype(str))

    event_encoder = LabelEncoder()
    merged['PREDICTED_EVENT_TYPE'] = event_encoder.fit_transform(
        merged['PREDICTED_EVENT_TYPE'].astype(str)
    )

    X = merged[features]
    y = merged[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)

    joblib.dump(model, "linre_model.pkl")
    joblib.dump(event_encoder, "linre_encoder.pkl")

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = targets)

    for col in clippers:
        y_pred_df[col] = np.maximum(y_pred_df[col], 0)
    y_pred_df['DAMAGE_CROPS'] = np.maximum(y_pred_df['DAMAGE_CROPS'], 0)

    y_pred_clipped = y_pred_df.to_numpy()
    print("\nAfter clipping, min values:")
    print(y_pred_df[clippers].min())

    print("MSE:", mean_squared_error(y_test, y_pred_clipped))
    # Mean Squared Error; how far off we were, avg, squared
    # we're a couple billion off, which is pretty good, all things considered
    print("R² score:", r2_score(y_test, y_pred_clipped, multioutput='uniform_average'))
    # Coefficient of Determination; how well our model explains variation in data
    # (1 is perfect, 0 is fully guessing)
    # so abt 30% of our targ outcome variability can be attributed to linre inputs (storm type, time, location, etc)
    results = pd.DataFrame(y_test)
    results["PREDICTED"] = y_pred.tolist()
    print(results.head())

    return model, event_encoder

def main ():
    wthr_df = load_data()
    knn_results = knn_mod(wthr_df)
    model, event_encoder = linre_mod(knn_results, wthr_df)
    print(event_encoder.classes_)

if __name__ == '__main__':
    main()