import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
from functions import Data2PPG #load_data
import matplotlib.pyplot as plt

train_path = "Task\Train"
validation_path = "Task\Validation"
test_path = "Task\Test"

print("Loading training data...")
X_train, y_train = Data2PPG.get_metrics(train_path)
print("Loading validation data...")
X_val, y_val = Data2PPG.get_metrics(validation_path)
print("Loading testing data...")
X_test, y_test = Data2PPG.get_metrics(test_path)

# Initializing XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')

# Define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate model
scores = cross_val_score(
    xgb_reg,
    X_train,
    y_train,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1
)
# Force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

# Defining a Hyperparameter Mesh
param_grid = {
    'n_estimators': [100, 300, 700, 1000],  # Number of trees
    'max_depth': [5, 7, 9, 11],             # Maximum Tree Depth
}

# Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_squared_error',
    verbose=1
)

# Train the model on training data
grid_search.fit(X_train, y_train)

# Get best params
print("BEST_PARAMS: ", grid_search.best_params_)

# Predict the model on testing data
y_pred = grid_search.predict(X_test)
# Get MSE on testing and training metrics
mse = mean_squared_error(y_test, y_pred)
# Get RMSE on testing and training metrics
rmse = np.sqrt(mse)
print("MSE: ", mse)
print("RMSE: ", rmse)

xgb.plot_tree(grid_search.best_estimator_.get_booster())
plt.title("Our XGBoost tree with Graphviz")
plt.show()

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Predicting data")
plt.text(0, 87, f'MAE: {scores.mean()}\nMSE: {mse}\nRMSE: {rmse}\nBest Params: {grid_search.best_params_}')
plt.legend()
plt.show()