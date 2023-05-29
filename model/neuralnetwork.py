########################################################################################################################
# IMPORTS
import random

import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import math
from sklearn.model_selection import GridSearchCV
import sys

# temporary
from sklearn.model_selection import train_test_split
import exploration.exploration as exp

########################################################################################################################

########################################################################################################################
# INFO
"""
https://scikit-learn.org/stable/modules/neural_networks_supervised.html#

Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function by training on a dataset,
where is the number of dimensions for input and is the number of dimensions for output. Given a set of features and a
target , it can learn a non-linear function approximator for either classification or regression. It is different
from logistic regression, in that between the input and the output layer, there can be one or more non-linear layers,
called hidden layers.

Class MLPRegressor implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation
function in the output layer, which can also be seen as using the identity function as activation function.
Therefore, it uses the square error as the loss function, and the output is a set of continuous values.

MLPRegressor also supports multi-output regression, in which a sample can have more than one target.

Algorithms: MLP trains using Stochastic Gradient Descent, Adam, or L-BFGS.
"""


########################################################################################################################

########################################################################################################################
# IMPLEMENTATION

# temporary
def get_vienna_data():
    vienna_weekdays = pd.read_csv('../datasets/vienna_weekdays.csv', sep=',')
    vienna_weekdays['weekend'] = False
    vienna_weekend = pd.read_csv('../datasets/vienna_weekends.csv', sep=',')
    vienna_weekend['weekend'] = True
    vienna = pd.concat([vienna_weekend, vienna_weekdays], ignore_index=True, sort=False)
    return vienna


def calculateErrors(y_test, y_hat):
    mae = mean_absolute_error(y_test, y_hat)
    rmse = math.sqrt(mean_squared_error(y_test, y_hat))
    medae = median_absolute_error(y_test, y_hat)
    return mae, rmse, medae


data = get_vienna_data()
data = exp.explore_and_clean(data)

X = data.drop(['realSum'], axis=1)  # .to_numpy()
y = data['realSum']  # .to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

'''
scikit learn neural network + gridsearch gives a looooot of warnings
# Define the hyperparameter grid Create a dictionary where the keys represent the hyperparameters you want to tune,
# and the values are the list of possible values for each hyperparameter.
param_grid = {
    'activation': ['identity', 'identity', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam']#,
    #'hidden_layer_sizes': [(1, 2, 3, 4, 5), (5, 4, 3, 2, 1), (2, 3, 2), (50, 50, 50)]
}
# create the regressor
regr = MLPRegressor(random_state=1, max_iter=500)  # .fit(X_train, y_train)
# perform grid search
grid_search = GridSearchCV(regr, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=3)
grid_search.fit(X_train, y_train)
# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_regressor = grid_search.best_estimator_
# make predictions on the test set using the best estimator
y_pred = best_regressor.predict(X_test)
y_hat = regr.predict(X_test)

mae_opt, rmse_opt, medae_opt = calculateErrors(y_test, y_pred)
mae_default, rmse_default, medae_default = calculateErrors(y_test, y_hat)
'''

# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# chatGPT says
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate some sample data
#X, y = make_regression(n_samples=100, n_features=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,)],  # Specify different hidden layer sizes
    'activation': ['relu', 'tanh'],  # Activation functions to try
    'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
}

# Create the MLPRegressor model
model = MLPRegressor(random_state=42, max_iter=500)

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters: ", grid_search.best_params_)

# Get the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
rmse = math.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
print("Best Model MSE: ", rmse)