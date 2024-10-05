import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalAveragePooling1D,
    Flatten,
    Dense,
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from functions import load_data


# Paths to the data directories
train_path = "Task/Train"
validation_path = "Task/Validation"
test_path = "Task/Test"

# Pre-process the dataset
print("Loading training data...")
X_train, y_train = load_data(train_path)
print("Loading validation data...")
X_val, y_val = load_data(validation_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)

# Reshape data for CNN input
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(125, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
model.add(AveragePooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
model.add(GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="linear"))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)

# Summary of the model
model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE (Loss): {test_loss}")
print(f"Test MAE (Metric): {test_mae}")

# Plot training & validation loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss (MSE)")
plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mean_absolute_error"], label="Train MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Val MAE")
plt.title("Model Metric")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()

plt.tight_layout()
plt.show()