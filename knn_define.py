from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_cleaning import load_data
import joblib

def knn_mod(df, user_input = None):
    features = [
        'BEGIN_MONTH',
        'BEGIN_DAY',
        'BEGIN_HOUR',
        'BEGIN_LAT',
        'BEGIN_LON'
    ]

    df = df.dropna(subset = features + ['EVENT_TYPE']).copy()

    X = df[features]
    y = df['EVENT_TYPE']
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    joblib.dump(knn, "knn_model.pkl")

    if user_input is not None:
        user_input = user_input.copy()
        prediction = knn.predict(user_input)
        return prediction[0]
    # would only return what kind of storm it thinks ur user input could be

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = knn.predict(X_test)
    # runs predictions for all instances in the database, accuracy / all of that

    test_results = X_test.copy()
    test_results['ACTUAL_EVENT_TYPE'] = y_test.values
    test_results['PREDICTED_EVENT_TYPE'] = y_pred

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return test_results

def main():
    wthr_df = load_data('list_of_storm_details.csv')
    event_pred = knn_mod(wthr_df)
    print(event_pred.head())

if __name__ == "__main__":
    main()