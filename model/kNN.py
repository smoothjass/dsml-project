import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

def kNN_tune_hyperparameters(X_train, y_train, X_test, y_test):
    # hyperparameters
    k = list(range(1, 20))  # all options for k that should be tried
    parameter_grid = {
        'n_neighbors': k,  # Possible values for the number of neighbors
        'weights': ['uniform', 'distance'],
        # Possible values for the weight function (uniform: all the same weight, distance: closer ones have higher influence)
        'p': [1, 2]  # Possible values for the power parameter (1 for Manhattan distance, 2 for Euclidean distance)
    }
    # algorithm = auto (automatically chosen based on dataset)
    # leaf_size: only for ball_tree and kd_tree algorithms

    # create regressor
    regressor = KNeighborsRegressor()
    # search for the best combination of hyperparameters
    grid_search = GridSearchCV(regressor, parameter_grid, cv=5, scoring='neg_mean_squared_error')
    # scoring chosen to minimize mse
    # 5-fold cross validation of data bc:
    # - reasonable computational cost,

    grid_search.fit(X_train, y_train)

    # find best version
    best_params = grid_search.best_params_
    print("Best params: ", best_params)
    best_regressor = grid_search.best_estimator_
    y_hat = best_regressor.predict(X_test)
    test_kNN_Regression(y_test, y_hat)
    return best_regressor

def kNN_getBestModel(feature_group):
    knn_regressor = None
    '''
    if feature_group == "all":
        knn_regressor = KNeighborsRegressor(n_neighbors= , weights=, p=)
        return knn_regressor
    elif feature_group == "spatial":
        knn_regressor = KNeighborsRegressor(n_neighbors=, weights=, p=)
    elif feature_group == "quality":
        knn_regressor = KNeighborsRegressor(n_neighbors=, weights=, p=)
    elif feature_group == "basic":
        knn_regressor = KNeighborsRegressor(n_neighbors=, weights=, p=)
    elif feature_group == "spatial_and_basic":
        knn_regressor = KNeighborsRegressor(n_neighbors=, weights=, p=)
    elif feature_group == "quality_and_basic":
        knn_regressor = KNeighborsRegressor(n_neighbors=, weights=, p=)
    else:
        print("Unknown feature group")
    '''
    return knn_regressor


def test_kNN_Regression(y_test, y_hat):

 mse = mean_squared_error(y_test, y_hat)
 mae = mean_absolute_error(y_test, y_hat)
 rmse = math.sqrt(mean_squared_error(y_test, y_hat))
 medae = median_absolute_error(y_test, y_hat)
 print("Mean Squared Error:", mse)
 print("Mean Absolute Error:", mae)
 print("Root Mean Squared Error:", rmse)
 print("Median Absolute Error:", medae)



'''
 use like this
X = vienna.drop('realSum', axis=1)

y = vienna['realSum']

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

y_pred = kNN.kNN_Regression(X_train, y_train, X_test)
kNN.test_kNN_Regression(y_test, y_pred)
'''


