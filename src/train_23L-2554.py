from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Build paths relative to the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_prices_23L-2554.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "house_price_model_23L-2554.pkl"


def main():
    print("Student ID: 23L-2554")
    print(f"Loading dataset from: {DATA_PATH}")

    # Load the housing dataset
    data = pd.read_csv(DATA_PATH)

    # Separate the input features and target column
    X = data.drop(columns=["price"])
    y = data["price"]

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.select_dtypes(exclude=["number"]).columns

    # Fill missing numerical values using the median
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Fill and encode categorical values
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Apply the appropriate processing to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )

    # Combine preprocessing and model training
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(random_state=42)
            )
        ]
    )

    # Divide the data into training and testing portions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training the Random Forest regression model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate the trained model
    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.4f}")

    # Create the output folder and save the complete pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)

    print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()