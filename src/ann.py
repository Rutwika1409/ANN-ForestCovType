# Import required libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import classification_report

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

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


x_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()

x_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

# ## Build ANN Model

# ### Architecture
# 1. Input Layer
# 2. Hidden Layers
# 3. Output Layer


# Creating the ANN Model
model = keras.Sequential([
    keras.layers.Input(shape=(x_train.shape[1],)),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(7, activation='softmax')
])

# #### 128 neurons --> learn pattern
# #### 64 neuron --> deeper learning
# #### Softmax --> multi class prediction
model.summary()

# Compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# #### Adam --> optimimzer
# #### sparse_categorical_crossentropy --> multiclass classification loss
# #### Accuracy --> evaluation metric


# Train the model
print("\nTraining the Model...\n")

# Fitting the model
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)


# #### epochs --> one learning cycle (forward + backward propagation)
# #### batch_size --> rows processed together
print("Evaluating the Model...")

# Model Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# Predict the test data
y_pred = model.predict(x_test)

# Converting the probabilities of prediction to binary outputs
y_pred_classes = y_pred.argmax(axis=1)

print(
    classification_report(
        y_test,
        y_pred_classes
    )
)
# Saving model 

print("\nSaving the Model...")


MODEL_PATH = os.path.join(
    MODELS_DIR,
    "forest_cover_ann.keras"
)

model.save(MODEL_PATH)

print(
    f"\nModel saved at:\n{MODEL_PATH}"
)



