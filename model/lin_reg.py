import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score


def linearRegressor(X_train, y_train):

    #X = vienna.drop('realSum', axis=1)
    #y = vienna['realSum']

    # split into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    regression_model = LinearRegression()

    # Fit the model using cross-validation
    scores = cross_val_score(regression_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # Convert the negative mean squared error scores to positive
    mse_scores = -scores

    # Calculate the mean and standard deviation of the MSE scores
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)

    # Train the model using the entire training set
    regression_model.fit(X_train, y_train)


    return regression_model
