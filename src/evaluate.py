import os
import pandas as pd
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "forest_cover_ann.keras"
)


x_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze()


print("\nLoading Model...\n")

model = keras.models.load_model(
    MODEL_PATH
)


print("Evaluating Model...\n")

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nGenerating Predictions...\n")

y_pred = model.predict(
    x_test,
    verbose=0
)

y_pred_classes = y_pred.argmax(
    axis=1
)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred_classes
    )
)

print(
    "\nConfusion Matrix:\n"
)

cm = confusion_matrix(
    y_test,
    y_pred_classes
)

print(cm)

cm_df = pd.DataFrame(cm)

cm_df.to_csv(
    os.path.join(
        MODELS_DIR,
        "confusion_matrix.csv"
    ),
    index=False
)

print(
    f"\nConfusion matrix saved to:\n"
    f"{os.path.join(MODELS_DIR, 'confusion_matrix.csv')}"
)

print(
    f"\nFinal Accuracy: "
    f"{accuracy_score(y_test, y_pred_classes) * 100:.2f}%"
)