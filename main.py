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
    # we'll just put y_hat in there for now because we don't have the other models yet
    y_hat_tree = y_test
    y_hat_knn = y_test
    y_hat_lin_regr = y_test
    y_hat_nn = nn.fitAndTuneRandomized(X_train, y_train, X_test)

    y_hats = [y_hat_tree, y_hat_knn, y_hat_lin_regr, y_hat_nn]
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

# step 5 fit models using training set + hyperparameter tuning including cross validation with random-/gridsearch for
# each model and feature combination: fit and tune hyperparameters, use 5-fold cross validation, return y_hat of the
# best model try each model with different feature combinations this is in a comment block because it is the step
# which takes a lot of time and resources. We did this before and hardcoded the best models in the getBestModel per
# featureGroup functions
"""predictions_all = fitAndTune(X_train, y_train, X_test, 'all')
predictions_spatial = fitAndTune(X_train_spatial, y_train_spatial, X_test_spatial, 'spatial')
predictions_quality = fitAndTune(X_train_quality, y_train_quality, X_test_quality, 'quality')
predictions_basic = fitAndTune(X_train_basic, y_train_basic, X_test_basic, 'basic')
predictions_spatial_and_basic = fitAndTune(X_train_spatial_and_basic, y_train_spatial_and_basic,
                                           X_test_spatial_and_basic, 'spatial_and_basic')
predictions_quality_and_basic = fitAndTune(X_train_quality_and_basic, y_train_quality_and_basic,
                                           X_test_quality_and_basic, 'quality_and_basic')
predictions = [predictions_all, predictions_spatial, predictions_quality, predictions_basic,
                predictions_spatial_and_basic, predictions_quality_and_basic]"""

# step 6
# get all the best models
nn_models = [[nn.getBestModel("all"), X_train, y_train, X_test],
             [nn.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial],
             [nn.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality],
             [nn.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic],
             [nn.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic],
             [nn.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic]]
"""tree_models = [tree.getBestModel("all"), 
               tree.getBestModel("spatial"), 
               tree.getBestModel("quality"), 
               tree.getBestModel("basic"), 
               tree.getBestModel("spatial_and_basic"), 
               tree.getBestModel("quality_and_basic")]
knn_models = [knn.getBestModel("all"), 
              knn.getBestModel("spatial"), 
              knn.getBestModel("quality"), 
              knn.getBestModel("basic"), 
              knn.getBestModel("spatial_and_basic"), 
              knn.getBestModel("quality_and_basic")]
lin_regr_models = [lin_regr.getBestModel("all"), 
                   lin_regr.getBestModel("spatial"), 
                   lin_regr.getBestModel("quality"), 
                   lin_regr.getBestModel("basic"), 
                   lin_regr.getBestModel("spatial_and_basic"), 
                   lin_regr.getBestModel("quality_and_basic")]"""
models = [nn_models]  # , tree_models, knn_models, lin_regr_models]
# collect all the predictions
predictions = pd.DataFrame()
for model in models:
    y_hats = []
    algorithm = model[0][0].__class__.__name__
    algorithms = [algorithm, algorithm, algorithm, algorithm, algorithm, algorithm]
    features = ["all", "spatial", "quality", "basic", "spatial_and_basic", "quality_and_basic"]
    for m in model:
        m[0].fit(m[1], m[2])
        y_hat = m[0].predict(m[3])
        y_hats.append(y_hat)
    dictionary = {'model': algorithms,
                  'features': features,
                  'y_hat': y_hats}
    prediction = pd.DataFrame(dictionary)
    predictions = pd.concat([predictions, prediction], axis=0)


# step 7
# performance evaluation with test set
evaluations = []
for index, row in predictions.iterrows():
    mae, rmse, medae = calculateErrors(y_test, row['y_hat'])
    evaluations.append([row['model'], row['features'], mae, rmse, medae])
evaluations = pd.DataFrame(evaluations, columns=["model", "features", "mae", "rmse", "medae"])

# step 8
# permutation importance - evaluate to which extent features contribute to the score


# step 9
# summarize results
