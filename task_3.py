import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,  mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import DMatrix, train
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

model = xgb.XGBRegressor(objective='reg:root_mean_squared_error', random_state=123)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
)
predictions = model.predict(X_test)

print(model.__dict__)

# Calculate MAE
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', mae)

# Calculate MSE
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Calculate RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)
print('Root Mean Squared Error:', rmse)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model.history["loss"], label="Train Loss (MSE)")
plt.plot(model.history["val_loss"], label="Val Loss (MSE)")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model.history["mean_absolute_error"], label="Train MAE")
plt.plot(model.history["val_mean_absolute_error"], label="Val MAE")
plt.title("Model Metric")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()

plt.tight_layout()
plt.show()