import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import joblib

# Define the target variable (what we want to predict) and the features
TARGET_COLUMN = 'discounted_price'
DATA_PATH = 'expanded_clothes_shoes.csv'
PREPROCESSOR_PATH = 'src/preprocessor.pkl'
PROCESSED_DATA_PATH = 'src/processed_data.csv'


def clean_and_convert_price(df, column):
    """Removes currency symbols and converts price columns to numeric."""
    # Fix E501 by breaking the line up
    df[column] = (
        df[column].astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    # Convert to numeric, coercing errors to NaN
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def create_preprocessing_pipeline(df):
    """
    Creates and fits a ColumnTransformer pipeline for preprocessing.
    """
    # 1. Identify feature types
    numerical_features = ['rating']  # 'discounted_price' will be the target

    # Identify relevant categorical features for encoding
    categorical_features = [
        'exclusive-new', 'brand', 'customer_review',
        'material_type', 'product_category'
    ]

    # 2. Create the transformers for each feature type
    # Impute missing numeric values with the median and scale them
    numerical_transformer = Pipeline(steps=[
        # Impute missing numeric values with the median of the training data
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Impute missing categorical values with a constant and apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        # Impute missing categorical values with a constant string ('missing')
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop all other irrelevant columns
    )

    # Fit the preprocessor to the data
    preprocessor.fit(df)

    return preprocessor


def run_preprocessing():
    """Main function to load data, preprocess, and save files."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    print("Running data cleaning and preprocessing...")

    # --- Step 1: Data Cleaning (Handling messy price/currency columns) ---
    df = clean_and_convert_price(df, 'price')
    df = clean_and_convert_price(df, 'original_price')

    # Drop rows where the TARGET_COLUMN is missing or zero after cleaning
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    # --- Step 2: Feature Engineering (Example: Price difference) ---
    df['price_diff'] = df['original_price'] - df['price']

    # --- Step 3: Train-Test Split (Before fitting the pipeline) ---
    # We split here to prevent data leakage during scaling/encoding
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Step 4: Create and Fit Preprocessing Pipeline ---
    preprocessor = create_preprocessing_pipeline(X_train)

    # --- Step 5: Transform Data ---
    # The preprocessor is fitted on X_train, then transforms both
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert processed arrays back to DataFrames for consistency
    # (Note: This is simplified. In a real project, you'd get feature names from the pipeline)
    X_train_processed = pd.DataFrame(X_train_processed)
    X_test_processed = pd.DataFrame(X_test_processed)

    # --- Step 6: Save Preprocessor and Processed Data ---
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Pre-processor saved to {PREPROCESSOR_PATH}")

    # Combine X and y for the train.py script to use later
    # For CI simplicity, we just print the shape.
    print(f"X_train processed shape: {X_train_processed.shape}")
    print(f"y_train shape: {y_train.shape}")


if __name__ == "__main__":
    # Ensure the src directory exists
    os.makedirs('src', exist_ok=True)
    run_preprocessing()