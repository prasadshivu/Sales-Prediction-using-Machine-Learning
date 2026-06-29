import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# category_encoders is needed for Target Encoding
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders is not installed.")

def main():

    print("Loading Dataset...")

    file_path = "SampleSuperstore.csv"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully")
    print(f"Rows : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    # ---------------------------------------
    # Handling Missing Values
    # ---------------------------------------
    print("\nHandling Missing Values...")

    # Artificially create missing values
    df.loc[0:20, 'Quantity'] = np.nan

    # Replace missing values with median
    imputer = SimpleImputer(strategy='median')
    df[['Quantity']] = imputer.fit_transform(df[['Quantity']])

    print("Missing values after imputation:",
          df['Quantity'].isnull().sum())

    print("\nFirst 10 Quantity values:")
    print(df['Quantity'].head(10))

    # ---------------------------------------
    # Target Encoding
    # ---------------------------------------
    if TargetEncoder is not None:

        print("\nApplying Target Encoding...")

        encoder = TargetEncoder()

        df['Segment_Encoded'] = encoder.fit_transform(
            df['Segment'],
            df['Sales']
        )

        print(df[['Segment', 'Sales',
                  'Segment_Encoded']].head())

    else:
        print("Target Encoder not installed.")

    # ---------------------------------------
    # Feature Selection
    # ---------------------------------------
    print("\nSelecting Best Features...")

    features = [
        'Quantity',
        'Discount',
        'Profit'
    ]

    X = df[features]
    y = df['Sales']

    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=2
    )

    selector.fit(X, y)

    best_features = X.columns[
        selector.get_support()
    ].tolist()

    print("Selected Features:")
    print(best_features)

    # ---------------------------------------
    # Train-Test Split
    # ---------------------------------------
    X = df[best_features]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTraining Data Size : {X_train.shape}")
    print(f"Testing Data Size  : {X_test.shape}")

    # ---------------------------------------
    # Train Linear Regression Model
    # ---------------------------------------
    print("\nTraining Model...")

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    # ---------------------------------------
    # Prediction Results
    # ---------------------------------------
    print("\nPrediction Results\n")

    actual_sales = y_test.values[:5]
    predicted_sales = prediction[:5]

    for i in range(5):

        print("----------------------------")
        print(f"Predicted Sales : {round(predicted_sales[i],2)}")
        print(f"Actual Sales    : {round(actual_sales[i],2)}")
        print(f"Difference      : {round(abs(actual_sales[i]-predicted_sales[i]),2)}")

    # ---------------------------------------
    # Model Evaluation
    # ---------------------------------------
    print("\nModel Performance")

    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print(f"Mean Absolute Error : {round(mae,2)}")
    print(f"R2 Score            : {round(r2,4)}")


if __name__ == "__main__":
    main()