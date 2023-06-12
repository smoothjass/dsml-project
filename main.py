########################################################################################################################
# IMPORTS
# our project files
import acquisition.acquisition as acq
import exploration.exploration as exp
import preprocessing.preprocessing as pre
import model.kNN as knn
# import model.decisionTree as tree
import model.neuralnetwork as nn
# other libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.inspection import permutation_importance
import math
import pandas as pd


########################################################################################################################
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


def collectPredictions(models):
    predictions = pd.DataFrame()
    for model in models:
        y_hats = []
        scores = []
        algorithm = type(model[0][0]).__name__
        algorithms = [algorithm, algorithm, algorithm, algorithm, algorithm, algorithm]
        features = ["all", "spatial", "quality", "basic", "spatial_and_basic", "quality_and_basic"]
        for m in model:
            m[0].fit(m[1], m[2])
            y_hat = m[0].predict(m[3])
            y_hats.append(y_hat)
            score = m[0].score(m[3], m[4])
            scores.append(score)
        dictionary = {'model': algorithms,
                      'features': features,
                      'y_hat': y_hats,
                      'score': scores}
        prediction = pd.DataFrame(dictionary)
        predictions = pd.concat([predictions, prediction], axis=0)
    return predictions


def calculateErrors(y_test, y_hat):
    mae = mean_absolute_error(y_test, y_hat)
    rmse = math.sqrt(mean_squared_error(y_test, y_hat))
    medae = median_absolute_error(y_test, y_hat)
    return mae, rmse, medae


def calculatePermutationImportance(models):
    perm_imps = pd.DataFrame()
    for model in models:
        algorithm = type(model[0][0]).__name__
        perm_imps_list = []
        for m in model:
            m[0].fit(m[1], m[2])
            r = permutation_importance(m[0], m[3], m[4])
            for i in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    perm_imp_list = [algorithm, m[5], m[1].columns[i], r.importances_mean[i], r.importances_std[i]]
                    perm_imps_list.append(perm_imp_list)
        perm_imp = pd.DataFrame(perm_imps_list, columns=['model', 'feature_group', 'feature', 'imp_mean', 'imp_std'])
        perm_imps = pd.concat([perm_imps, perm_imp], axis=0)
    return perm_imps


########################################################################################################################
# MACHINE LEARNING WORKFLOW

# STEP 1
# get the labeled data from the .csv files (combines weekend and weekday data)
vienna = acq.get_vienna_data()

# STEP 2
# explore the data (plots etc.) and clean it
# (remove outliers, check class imbalance, check encoding, dimensionality reduction of unnecessary columns)
vienna = exp.explore_and_clean(vienna)

# STEP 3
# preprocess the data (normalize, scale)
vienna = pre.preprocess_data(vienna)

# STEP 4
# data splitting into training(, validation) and test sets for
# all features
# spatial features
# quality features
# basic features
# spatial and basic features
# quality and basic features
X = vienna.drop(['realSum'], axis=1)  # .to_numpy()
y = vienna['realSum']  # .to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_spatial = X[['lat', 'lng', 'dist', 'metro_dist', 'rest_index', 'attr_index']]
X_train_spatial, X_test_spatial, y_train_spatial, y_test_spatial = train_test_split(X_spatial, y, random_state=42)

X_quality = X[['cleanliness_rating', 'guest_satisfaction_overall', 'weekend']]
X_train_quality, X_test_quality, y_train_quality, y_test_quality = train_test_split(X_quality, y, random_state=42)

X_basic = X[['host_is_superhost', 'room_private', 'person_capacity', 'bedrooms', 'entire_home_apt']]
X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X_basic, y, random_state=42)

X_spatial_and_basic = pd.concat([X_spatial, X_basic], axis=1)
X_train_spatial_and_basic, X_test_spatial_and_basic, y_train_spatial_and_basic, y_test_spatial_and_basic = \
    train_test_split(X_spatial_and_basic, y, random_state=42)

X_quality_and_basic = pd.concat([X_quality, X_basic], axis=1)
X_train_quality_and_basic, X_test_quality_and_basic, y_train_quality_and_basic, y_test_quality_and_basic = \
    train_test_split(X_quality_and_basic, y, random_state=42)

# STEP 5
# fit models using training set + hyperparameter tuning including cross validation with random-/gridsearch for
# each model and feature combination: this is in a comment block because it is the step which takes a lot of time and
# resources. We did this before and hardcoded the best models in the getBestModel per featureGroup functions
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

# STEP 6
# get all the best models and predictions
nn_models = [[nn.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [nn.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [nn.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [nn.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [nn.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [nn.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]
"""tree_models = [[tree.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [tree.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [tree.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [tree.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [tree.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [tree.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]"""
knn_models = [[knn.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [knn.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [knn.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [knn.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [knn.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [knn.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]
"""lin_regr_models = [[nn.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [lin_regr.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [lin_regr.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [lin_regr.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [lin_regr.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [lin_regr.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]"""
models = [nn_models, knn_models]  # , tree_models, lin_regr_models]
# collect all the predictions
predictions = collectPredictions(models)

# step 7
# performance evaluation with test set
evaluations = []
for index, row in predictions.iterrows():
    mae, rmse, medae = calculateErrors(y_test, row['y_hat'])
    evaluations.append([row['model'], row['features'], mae, rmse, medae, row['score']])
    del mae, rmse, medae
evaluations = pd.DataFrame(evaluations, columns=["model", "features", "mae", "rmse", "medae", "score"])
# note that the algorithms have different score methods e.g. R^2 for MLPRegressor and accuracy for KNN

# step 8
# permutation importance - evaluate to which extent features contribute to the score
permutation_importance = calculatePermutationImportance(models)

# step 9
# summarize results
# tidy up to see the results better in the SciView
del X_basic, X_spatial, X_quality, X_spatial_and_basic, X_quality_and_basic, \
    X_test_basic, X_test_spatial, X_test_quality, X_test_spatial_and_basic, X_test_quality_and_basic, \
    X_train_basic, X_train_spatial, X_train_quality, X_train_spatial_and_basic, X_train_quality_and_basic, \
    y_test_basic, y_test_spatial, y_test_quality, y_test_spatial_and_basic, y_test_quality_and_basic, \
    y_train_basic, y_train_spatial, y_train_quality, y_train_spatial_and_basic, y_train_quality_and_basic
