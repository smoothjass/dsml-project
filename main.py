########################################################################################################################
# IMPORTS
# our project files
import acquisition.acquisition as acq
import exploration.exploration as exp
import preprocessing.preprocessing as pre
# import model.kNN as kNN
# import model.decisionTree as tree
import model.neuralnetwork as nn
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


def fitAndTune(X_train, y_train, X_test, features):
    # y_hat_tree =
    # y_hat_knn =
    # y_hat_lin_regr =
    # y_hat_nn = nn.fitAndTuneRandomized(X_train, y_train, X_test)
    y_hat_nn = nn.predict(X_train, y_train, X_test)

    y_hats = [y_test, y_test, y_test, y_hat_nn]
    models = ['tree', 'knn', 'lin_regr', 'nn']
    feature_group = [features, features, features, features]
    dictionary = {'model': models,
                  'y_hat': y_hats,
                  'features': feature_group}
    predictions = pd.DataFrame(dictionary)
    return predictions


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
# feature selection and splitting
# spatial features
X_spatial = X[['lat', 'lng', 'dist', 'metro_dist', 'rest_index', 'attr_index']]
X_train_spatial, X_test_spatial, y_train_spatial, y_test_spatial = train_test_split(X_spatial, y, random_state=42)
# quality features
X_quality = X[['cleanliness_rating', 'guest_satisfaction_overall', 'weekend']]
X_train_quality, X_test_quality, y_train_quality, y_test_quality = train_test_split(X_quality, y, random_state=42)
# basic features
X_basic = X[['host_is_superhost', 'room_private', 'person_capacity', 'bedrooms', 'entire_home_apt']]
X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X_basic, y, random_state=42)
# combinations of features
X_spatial_and_basic = pd.concat([X_spatial, X_basic], axis=1)
X_train_spatial_and_basic, X_test_spatial_and_basic, y_train_spatial_and_basic, y_test_spatial_and_basic = \
    train_test_split(X_spatial_and_basic, y, random_state=42)
X_quality_and_basic = pd.concat([X_quality, X_basic], axis=1)
X_train_quality_and_basic, X_test_quality_and_basic, y_train_quality_and_basic, y_test_quality_and_basic = \
    train_test_split(X_quality_and_basic, y, random_state=42)

# step 5 fit models using training set + hyperparameter tuning including cross validation with random-/gridsearch
# todo - for each model: a function which takes training set (X and y) and X_test as input, performs fit and
#  hyperparameter tuning with cross validation, stores best model for future use and returns y_hat
# todo - try each model with different feature combinations
predictions_all = fitAndTune(X_train, y_train, X_test, 'all')
predictions_spatial = fitAndTune(X_train_spatial, y_train_spatial, X_test_spatial, 'spatial')
predictions_quality = fitAndTune(X_train_quality, y_train_quality, X_test_quality, 'quality')
predictions_basic = fitAndTune(X_train_basic, y_train_basic, X_test_basic, 'basic')
predictions_spatial_and_basic = fitAndTune(X_train_spatial_and_basic, y_train_spatial_and_basic,
                                           X_test_spatial_and_basic, 'spatial_and_basic')
predictions_quality_and_basic = fitAndTune(X_train_quality_and_basic, y_train_quality_and_basic,
                                           X_test_quality_and_basic, 'quality_and_basic')
predictions = [predictions_all, predictions_spatial, predictions_quality, predictions_basic,
                predictions_spatial_and_basic, predictions_quality_and_basic]

# step 6
# performance evaluation with test set
evaluations = pd.DataFrame()
for df in predictions:
    evaluation = []
    for index, row in df.iterrows():
        mae, rmse, medae = calculateErrors(y_test, row['y_hat'])
        evaluation.append([row['model'], row['features'], mae, rmse, medae])
    evaluation = pd.DataFrame(evaluation, columns=["model", "features", "mae", "rmse", "medae"])
    evaluations = pd.concat([evaluations, evaluation], axis=0)

# step 7
# summarize results
