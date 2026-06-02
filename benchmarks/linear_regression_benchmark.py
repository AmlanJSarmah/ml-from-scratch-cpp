import sys
import pandas as pd

from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    if df.shape[1] < 2:
        sys.exit(1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if not pd.api.types.is_numeric_dtype(y):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Keeping your original output format
    accuracy = r2

    print(accuracy)
    print(r2)
    print(rmse)
    print(mae)
    print(mse)

if __name__ == "__main__":
    main()