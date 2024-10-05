import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RepeatedKFold
from functions import load_data
import pandas as pd
import matplotlib.pyplot as plt

# Paths to the data directories
train_path = "Task/Train"
validation_path = "Task/Validation"
test_path = "Task/Test"

print("Loading training data...")
X_train, y_train = load_data(train_path)  # Используем длину сигнала 5 секунд
print("Loading validation data...")
X_val, y_val = load_data(validation_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)

# Инициализация XGBoost регрессора
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgb_reg, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

# Определение сетки гиперпараметров
param_grid = {
    'n_estimators': [100, 300, 700, 1000],  # Количество деревьев
    'max_depth': [5, 7, 9, 11],            # Максимальная глубина деревьев
}

# Grid Search с кросс-валидацией
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=1)

# Обучение модели на тренировочных данных
grid_search.fit(X_train, y_train)

print("BEST_PARAMS: ", grid_search.best_params_)
print(grid_search.best_score_)

y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = r2_score(y_test, y_pred)

print("MSE: ", mse)
print("RMSE: ", rmse)

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Predicting data")
plt.legend()
plt.show()