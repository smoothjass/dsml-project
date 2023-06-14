########################################################################################################################
# IMPORTS

# our project files
import acquisition.acquisition as acq
import exploration.exploration as exp
import preprocessing.preprocessing as pre
import model.kNN as knn
import model.decisionTree as tree
import model.neuralnetwork as nn
import model.lin_reg as lin_regr

# other libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.inspection import permutation_importance
import math
import pandas as pd


########################################################################################################################
def fitAndTune(X_train, y_train, X_test, features):
    # todo anpassen
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
            '''importances_mean: ndarray of shape (n_features, )
                Mean of feature importance over n_repeats.
                importances_std: ndarray of shape (n_features, )
                Standard deviation over n_repeats.'''
            for i in r.importances_mean.argsort()[::-1]:
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    perm_imp_list = [algorithm, m[5], m[1].columns[i], r.importances_mean[i], r.importances_std[i]]
                    perm_imps_list.append(perm_imp_list)
        perm_imp = pd.DataFrame(perm_imps_list, columns=['model', 'feature_group', 'feature', 'imp_mean', 'imp_std'])
        perm_imps = pd.concat([perm_imps, perm_imp], axis=0)
    return perm_imps


########################################################################################################################
# MACHINE LEARNING WORKFLOW

# STEP 1 DATA ACQUISITION
# get the labeled data from the .csv files (combines weekend and weekday data)
# the data can be downloaded and read up on here:
# https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities?select=vienna_weekdays.csv
# https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities?select=vienna_weekends.csv
# put these two csv files in a folder namend 'datasets' within the project folder in able to use acq.get_vienna_data()
vienna = acq.get_vienna_data()

# STEP 2 DATA EXPLORATION AND CLEANING
# explore the data (plots etc.) and clean it
# (remove outliers, check class imbalance, check encoding, dimensionality reduction of unnecessary columns)
vienna = exp.explore_and_clean(vienna)

# STEP 3 DATA PREPROCESSING
# preprocess the data (scale)
# we tried standardScaler and MinMaxScaler
# standardScaler performed better overall
vienna = pre.preprocess_data(vienna)

# STEP 4 MODEL BUILDING
# STEP 4.1 splits
# data splitting into training(, validation) and test sets for
# all features
# spatial features
# quality features
# basic features
# spatial and basic features
# quality and basic features
# as we use GridSearchCV or RandomSearchCV in the models training, we don't need to split off extra validation sets here
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

# STEP 4.2 hp tuning + cross validation for different feature groups
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

# STEP 4.3 make predictions for evaluation
# get all the best models and predictions
nn_models = [[nn.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [nn.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [nn.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [nn.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [nn.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [nn.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]
tree_models = [[tree.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [tree.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [tree.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [tree.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [tree.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [tree.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]
knn_models = [[knn.getBestModel("all"), X_train, y_train, X_test, y_test, "all"],
             [knn.getBestModel("spatial"), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [knn.getBestModel("quality"), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [knn.getBestModel("basic"), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [knn.getBestModel("spatial_and_basic"), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [knn.getBestModel("quality_and_basic"), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]
lin_regr_models = [[lin_regr.linearRegressor(X_train, y_train), X_train, y_train, X_test, y_test, "all"],
             [lin_regr.linearRegressor(X_train_spatial, y_train_spatial), X_train_spatial, y_train_spatial, X_test_spatial, y_test_spatial, "spatial"],
             [lin_regr.linearRegressor(X_train_quality, y_train_quality), X_train_quality, y_train_quality, X_test_quality, y_test_quality, "quality"],
             [lin_regr.linearRegressor(X_train_basic, y_train_basic), X_train_basic, y_train_basic, X_test_basic, y_test_basic, "basic"],
             [lin_regr.linearRegressor(X_train_spatial_and_basic, y_train_spatial_and_basic), X_train_spatial_and_basic, y_train_spatial_and_basic,
              X_test_spatial_and_basic, y_test_spatial_and_basic, "spatial+basic"],
             [lin_regr.linearRegressor(X_train_quality_and_basic, y_train_quality_and_basic), X_train_quality_and_basic, y_train_quality_and_basic,
              X_test_quality_and_basic, y_test_quality_and_basic, "quality+basic"]]
models = [nn_models, knn_models, tree_models, lin_regr_models]
# collect all the predictions
predictions = collectPredictions(models)

# STEP 4.4 calculate error metrics
# performance evaluation with test set
# we calculate the mean absolute error (MAE), root mean squared error (RMSE) and median absolute error (MedAE) and
# the score of the model (note! different score methods for different algorithms) and collect them in a dataframe
evaluations = []
for index, row in predictions.iterrows():
    mae, rmse, medae = calculateErrors(y_test, row['y_hat'])
    evaluations.append([row['model'], row['features'], mae, rmse, medae, row['score']])
    del mae, rmse, medae
evaluations = pd.DataFrame(evaluations, columns=["model", "features", "mae", "rmse", "medae", "score"])
# important that the algorithms have different score methods e.g. R^2 for MLPRegressor and accuracy for KNN
# can be found in the scikit learn documentation of the algorithms

# STEP 4.5 permutation importance
# evaluate to which extent features contribute to the score
# i.e. how much the score decreases when a single feature is shuffled
# these results are also collected in a dataframe for comparison
permutation_importance = calculatePermutationImportance(models)

# STEP 5 INTERPRETATION AND DISCUSSION
# summarize results
# the dataframes evaluations and permutation_importance contain the most significant summaries
# best models:
# knn and tree with spatial and basic features (are they overfit though?)
# linear regression and nn worked best when using all features
# most important features
# linear regression: entire_home and room_private most important
# nn: distance to city center, distance to metro and restaurant index most important
# tree: attraction index, longitude, distance to center most important
# knn: person capacity and bedrooms most important (in the model which trained on spatial and basic dataset)
# knn: longitude and distance to center most important (in the model which trained on spatial only dataset)

# once the hp optimization is done, the best model can easily be trained on the data and deployed e.g. on a website,
# however the hp tuning and choosing of the algorithm are quite time-consuming.
# as we were able to predict the prices correctly within a range of roughly 10€ (median squared error) or 50€ (root mean
# squared error) the model seems to be good enough from a domain expert point of view, however none of us are domain
# experts, and we'd have to get some expert's opinions on this.
# from a societal point of view, this model could help avoid unfair increases in prices or unfair pricing compared to
# competitors in general. It could also help people who are just getting started as hosts and don't know how to price
# their bnb.

# tidy up a bit to see the results better in the SciView
del X_basic, X_spatial, X_quality, X_spatial_and_basic, X_quality_and_basic, \
    X_test_basic, X_test_spatial, X_test_quality, X_test_spatial_and_basic, X_test_quality_and_basic, \
    X_train_basic, X_train_spatial, X_train_quality, X_train_spatial_and_basic, X_train_quality_and_basic, \
    y_test_basic, y_test_spatial, y_test_quality, y_test_spatial_and_basic, y_test_quality_and_basic, \
    y_train_basic, y_train_spatial, y_train_quality, y_train_spatial_and_basic, y_train_quality_and_basic
