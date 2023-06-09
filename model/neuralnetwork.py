########################################################################################################################
# IMPORTS
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
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

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

Randomized search on hyper parameters.
The parameters of the estimator used to apply these methods are optimized by cross-validated search over parameter 
settings.
In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings 
is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter."""
########################################################################################################################

########################################################################################################################
# IMPLEMENTATION
def fitAndTune(X_train, y_train, X_test):
    y_hat = None
    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'hidden_layer_sizes': [(1,), (2,), (3,), (5,), (7,), (10,),
                               (15,), (17,), (19,), (20,), (21,), (23,), (25,)],  # Specify different hidden layer sizes
        'activation': ['relu', 'tanh'],  # Activation functions to try
        'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
    }
    # Create the MLPRegressor model
    # lbfgs converges faster
    model = MLPRegressor(random_state=42, max_iter=2000, solver='lbfgs')

    # Create the RandomizedSearchCV object
    # default 5-fold cross validation with shuffle_False
    random_search = RandomizedSearchCV(model, param_grid, verbose=3, n_iter=20)

    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best Hyperparameters: ", random_search.best_params_)

    # Get the best model found by RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predict y_hat using the best model
    y_hat = best_model.predict(X_test)

    return y_hat
########################################################################################################################

# model = MLPRegressor(random_state=42, max_iter=2000)
# Best Hyperparameters:  {'hidden_layer_sizes': (20,), 'alpha': 0.001, 'activation': 'relu'}
# nn,51.73184,78.34428,37.00069
# some did not converge
# param_grid = {
#         'hidden_layer_sizes': [(1,), (2,), (3,), (10,), (15,), (20,)],  # Specify different hidden layer sizes
#         'activation': ['relu', 'tanh'],  # Activation functions to try
#         'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
#     }

# model = MLPRegressor(random_state=42, max_iter=2000)
# Best Hyperparameters:  {'hidden_layer_sizes': (25,), 'alpha': 0.01, 'activation': 'relu'}
# nn,51.83142,77.98413,37.25985
# some did not converge
# param_grid = {
#         'hidden_layer_sizes': [(15,), (17,), (19,), (20,),
#         (21,), (23,), (25,)],  # Specify different hidden layer sizes
#         'activation': ['relu', 'tanh'],  # Activation functions to try
#         'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
#     }

# model = MLPRegressor(random_state=42, max_iter=2000, solver='lbfgs')
# Best Hyperparameters:  {'hidden_layer_sizes': (7,), 'alpha': 0.0001, 'activation': 'relu'}
# nn,51.00826,75.35174,36.07388
# param_grid = {
#         'hidden_layer_sizes': [(1,), (2,), (3,), (5,), (7,), (10,), (15,), (17,), (19,),
#         (20,), (21,), (23,), (25,)],  # Specify different hidden layer sizes
#         'activation': ['relu', 'tanh'],  # Activation functions to try
#         'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
#     }
