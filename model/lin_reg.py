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

    # Make predictions on the test set
    #y_hat = regression_model.predict(X_test)

    #r2 = r2_score(y_test, y_hat)
    #mae = mean_absolute_error(y_test, y_hat)

    #print("Mean Squared Error (CV):", mean_mse)
    #print("Standard Deviation of MSE (CV):", std_mse)
    #print("R-squared:", r2)
    #print("Mean Absolute Error:", mae)

    return regression_model

'''
def testLinReg(y_hat, y_test):
    mse = mean_squared_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    mae = mean_absolute_error(y_test, y_hat)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("Mean Absolute Error:", mae)
'''
