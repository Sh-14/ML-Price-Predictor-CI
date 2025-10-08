import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# --- Configuration ---
MODEL_PATH = 'src/best_ml_model.pkl'
PREPROCESSOR_PATH = 'src/preprocessor.pkl'
DATA_PATH = 'expanded_clothes_shoes.csv'
TARGET_COLUMN = 'discounted_price'

# CRITICAL: This threshold determines CI success. Adjust this value
# based on your model's ACTUAL performance.
MAE_THRESHOLD = 2000


def clean_and_convert_price(df, column):
    """Helper to clean price columns, fixing E501 and W291."""
    df[column] = (
        df[column].astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def run_ci_tests():
    """Loads the model and runs a critical performance test."""
    print("--- Starting CI Model Verification Test (MAE check) ---")

    # 1. Check for required artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}. Was train.py run?")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Error: Preprocessor not found at {PREPROCESSOR_PATH}.")

    # 2. Load the artifacts
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and Preprocessor loaded successfully.")

    # 3. Prepare the Test Data (Must exactly match the split done during training)
    df = pd.read_csv(DATA_PATH)
    df = clean_and_convert_price(df, 'price')
    df = clean_and_convert_price(df, 'original_price')
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Get the test set (using same random_state=42 as in train.py)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Transform Test Data
    X_test_processed = preprocessor.transform(X_test)

    # 5. Run the Core Performance Test
    y_pred = model.predict(X_test_processed)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Calculated Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Acceptable MAE Threshold: {MAE_THRESHOLD:.2f}")

    # Assertion: If the calculated MAE is GREATER than the threshold, the test FAILS.
    assert mae <= MAE_THRESHOLD, (
        f"TEST FAILED: Model MAE ({mae:.2f}) is WORSE than the allowed threshold ({MAE_THRESHOLD:.2f})."
    )

    print("\n✅ CI TEST PASSED: Model performance meets the required threshold.")


if __name__ == "__main__":
    run_ci_tests()
