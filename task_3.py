import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
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

# Объединение тренировочных и валидационных данных для общей тренировки с последующей кросс-валидацией
X_train_full = np.concatenate((X_train, X_val), axis=0)
y_train_full = np.concatenate((y_train, y_val), axis=0)

# Разделение данных на тренировочные и тестовые
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Инициализация XGBoost регрессора
xgb_reg = xgb.XGBRegressor()

# Определение сетки гиперпараметров
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],  # Количество деревьев
    'max_depth': [3, 5, 7, 9, 11],            # Максимальная глубина деревьев
}

# Grid Search с кросс-валидацией
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

# Обучение модели на тренировочных данных
grid_search.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

# Получение лучших параметров
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

eval_res = grid_search.best_estimator_.evals_result()
print("EVALS_RES: ", eval_res)

print("HISTORY: ", grid_search.history)

# Оценка модели на валидационных данных
y_val_pred = grid_search.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation Mean Squared Error: {val_mse}")

# Оценка модели на тестовых данных
y_test_pred = grid_search.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean Squared Error: {test_mse}")

plt.subplot(1, 2, 1)
pd.DataFrame({
    'train': eval_res['validation_0']['rmse'],
    'valid': eval_res['validation_1']['rmse']
}).plot()
plt.xlabel('boosting round')
plt.ylabel('objective')
plt.tight_layout()
plt.show()

# Теперь можно провести дополнительные эксперименты, изменяя длину сигнала
for signal_length in [3, 7, 10]:  # Изучение других длин сигнала
    print(f"\nEvaluating with signal length of {signal_length} seconds...")
    X_train, y_train = load_data(train_path, signal_length=signal_length)
    X_val, y_val = load_data(validation_path, signal_length=signal_length)
    X_test, y_test = load_data(test_path, signal_length=signal_length)

    # Обучаем модель на новом наборе данных
    grid_search.fit(X_train, y_train)
    
    # Оценка на валидационных данных
    y_val_pred = grid_search.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"Validation MSE for {signal_length} seconds: {val_mse}")
    
    # Оценка на тестовых данных
    y_test_pred = grid_search.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE for {signal_length} seconds: {test_mse}")