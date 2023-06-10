########################################################################################################################
# IMPORTS
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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
def fitAndTuneRandomized(X_train, y_train, X_test):
    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
                               (11,), (12,), (13,), (14,), (15,)],
        # Specify different hidden layer sizes
        'activation': ['relu', 'tanh'],  # Activation functions to try
        'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
    }
    # Create the MLPRegressor model
    # lbfgs converges faster
    model = MLPRegressor(random_state=42, max_iter=2000, solver='lbfgs')

    # Create the RandomizedSearchCV object
    # default 5-fold cross validation with shuffle_False
    random_search = RandomizedSearchCV(model, param_grid, verbose=3, n_iter=50)

    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best Hyperparameters: ", random_search.best_params_)

    # Get the best model found by RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predict y_hat using the best model
    y_hat = best_model.predict(X_test)

    return y_hat


def fitAndTuneGrid(X_train, y_train, X_test):
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
                               (11,), (12,), (13,), (14,), (15,)],
        # Specify different hidden layer sizes
        'activation': ['relu', 'tanh'],  # Activation functions to try
        'alpha': [0.0001, 0.001, 0.01]  # Regularization parameter
    }
    # Create the MLPRegressor model
    # lbfgs converges faster
    model = MLPRegressor(random_state=42, max_iter=3000, solver='lbfgs')

    # Create the RandomizedSearchCV object
    # default 5-fold cross validation with shuffle_False
    grid_search = GridSearchCV(model, param_grid, verbose=3)

    # Fit the RandomizedSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best Hyperparameters: ", grid_search.best_params_)
    # Best Hyperparameters: {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (13,)}

    # Get the best model found by RandomizedSearchCV
    best_model = grid_search.best_estimator_

    # Predict y_hat using the best model
    y_hat = best_model.predict(X_test)

    return y_hat

# once the number 13 was found as the best hidden layer size, we tried tuning the other parameters but the
# improvement was not significant
def fitAndTuneActivationAndAlpha(X_train, y_train, X_test):
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'activation': ['relu', 'tanh', 'logistic', 'identity'],  # Activation functions to try
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]  # Regularization parameter
    }
    # Create the MLPRegressor model
    # lbfgs converges faster
    model = MLPRegressor(random_state=42, max_iter=3000, solver='lbfgs', hidden_layer_sizes=(13,))

    # Create the RandomizedSearchCV object
    # default 5-fold cross validation with shuffle_False
    grid_search = GridSearchCV(model, param_grid, verbose=3)

    # Fit the RandomizedSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters found
    print("Best Hyperparameters: ", grid_search.best_params_)

    # Get the best model found by RandomizedSearchCV
    best_model = grid_search.best_estimator_

    # Predict y_hat using the best model
    y_hat = best_model.predict(X_test)

    return y_hat


def predict(X_train, y_train, X_test):
    model_all = MLPRegressor(random_state=42, max_iter=3000, solver='lbfgs', hidden_layer_sizes=(13,), activation='relu',
                         alpha=0.0001)

    # Fit the model to the training data
    model_all.fit(X_train, y_train)

    # Predict y_hat using the trained model
    y_hat = model_all.predict(X_test)

    return y_hat
########################################################################################################################

# Best Hyperparameters:  {'hidden_layer_sizes': (13,), 'alpha': 0.0001, 'activation': 'relu'} for all
# Best Hyperparameters:  {'hidden_layer_sizes': (13,), 'alpha': 0.0001, 'activation': 'relu'} for spatial
# Best Hyperparameters:  {'hidden_layer_sizes': (7,), 'alpha': 0.0001, 'activation': 'relu'} for quality
# Best Hyperparameters:  {'hidden_layer_sizes': (14,), 'alpha': 0.0001, 'activation': 'relu'} for basic
# Best Hyperparameters:  {'hidden_layer_sizes': (12,), 'alpha': 0.01, 'activation': 'relu'} for spatial and basic
# Best Hyperparameters:  {'hidden_layer_sizes': (10,), 'alpha': 0.001, 'activation': 'relu'} for quality and basic
