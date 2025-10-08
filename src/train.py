import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os

# --- File Paths ---
DATA_PATH = 'expanded_clothes_shoes.csv'
PREPROCESSOR_PATH = 'src/preprocessor.pkl'
MODEL_PATH = 'src/best_ml_model.pkl'

# Target column for prediction
TARGET_COLUMN = 'discounted_price'


def clean_and_convert_price(df, column):
    """Helper to clean price columns, matching logic in preprocess.py and train.py."""
    df[column] = (
        df[column].astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def load_data_and_preprocessor():
    """Loads the raw data and the fitted preprocessor pipeline."""
    if not os.path.exists(DATA_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print(f"Error: Required files not found. Check if {DATA_PATH} and {PREPROCESSOR_PATH} exist.")
        return None, None

    # Load raw data (needed to extract X and y before processing)
    df = pd.read_csv(DATA_PATH)

    # Simple cleaning needed to identify the target and drop NaNs
    df = clean_and_convert_price(df, 'price')
    df = clean_and_convert_price(df, 'original_price')
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split data to match the split done during preprocessor fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load the fitted preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Apply the preprocessor to the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Note: We keep y_train/y_test as pandas Series for easy evaluation
    return X_train_processed, X_test_processed, y_train, y_test


def train_model(X_train, y_train):
    """Initializes and trains the Linear Regression model."""
    print("Starting model training...")

    # Define the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


def evaluate_and_save(model, X_test, y_test):
    """Evaluates the model on the test set and saves it."""
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print("\nModel Performance (Test Set):")  # Evolved from f-string fix
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data_and_preprocessor()

    if X_train is not None:
        trained_model = train_model(X_train, y_train)
        evaluate_and_save(trained_model, X_test, y_test)