import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "covtype.csv"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    df.drop_duplicates(inplace=True)

    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train - 1
    y_test = y_test - 1
    print("Target labels:\n")
    print(sorted(y_train.unique()))
    pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    ).to_csv(
        os.path.join(DATA_DIR, "X_train.csv"),
        index=False
    )

    pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    ).to_csv(
        os.path.join(DATA_DIR, "X_test.csv"),
        index=False
    )

    pd.DataFrame(
        y_train
    ).to_csv(
        os.path.join(DATA_DIR, "y_train.csv"),
        index=False
    )

    pd.DataFrame(
        y_test
    ).to_csv(
        os.path.join(DATA_DIR, "y_test.csv"),
        index=False
    )

    with open(
        os.path.join(MODELS_DIR, "scaler.pkl"),
        "wb"
    ) as f:
        pickle.dump(scaler, f)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()