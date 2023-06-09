########################################################################################################################
# IMPORTS
# our project files
import acquisition.acquisition as acq
import exploration.exploration as exp
import preprocessing.preprocessing as pre
# import model.kNN as kNN
# import model.decisionTree as tree
# import model.neuralnetwork as nn
# other libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import math
import pandas as pd


########################################################################################################################

def calculateErrors(y_test, y_hat):
    mae = mean_absolute_error(y_test, y_hat)
    rmse = math.sqrt(mean_squared_error(y_test, y_hat))
    medae = median_absolute_error(y_test, y_hat)
    return mae, rmse, medae


# Machine Learning Workflow

# step 1
# get the labeled data from the .csv files (combines weekend and weekday data)
vienna = acq.get_vienna_data()

# step 2
# explore the data (plots etc.) and clean it
# (remove outliers, check class imbalance, check encoding, dimensionality reduction of unnecessary columns)
vienna = exp.explore_and_clean(vienna)

# step 3
# preprocess the data (normalize, scale)
vienna = pre.preprocess_data(vienna)

# step 4
# data splitting into training(, validation) and test sets
X = vienna.drop(['realSum'], axis=1)  # .to_numpy()
y = vienna['realSum']  # .to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# step 5 fit models using training set + hyperparameter tuning including cross validation with random-/gridsearch
# todo for each model: a function which takes training set (X and y) as input, performs fit and hyperparameter tuning
#  with cross validation, stores best model for future use and returns y_hat
'''
y_hat_tree = 
y_hat_knn = 
y_hat_lin_regr = 
y_hat_nn = 
predictions = [y_hat_tree, y_hat_knn, y_hat_lin_regr, y_hat_nn] 
'''
y_hats = [y_test, y_test, y_test, y_test]
models = ['tree', 'knn', 'lin_regr', 'nn']
dictionary = {'model': models,
              'y_hat': y_hats}
predictions = pd.DataFrame(dictionary)

# step 6
# performance evaluation with test set
performance = []
for index, row in predictions.iterrows():
    mae, rmse, medae = calculateErrors(y_test, row['y_hat'])
    performance.append([row['model'], mae, rmse, medae])
performance = pd.DataFrame(performance, columns=["model", "mae", "rmse", "medae"])

# step 7
# summarize results
