from matplotlib import pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import preprocessing.preprocessing as pre
import acquisition.acquisition as acq
import exploration.exploration as exp

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

df = vienna.copy()

# all columns except 'realSum' are features
# X = df.drop(columns=['realSum'])
# y = df['realSum']

# only the spatial features ('lat', 'lng', 'dist', 'metro_dist', 'rest_index', 'attr_index') are used as features
# X = df[['lat', 'lng', 'dist', 'metro_dist', 'rest_index', 'attr_index']]
# y = df['realSum']

# only the quality features ('cleanliness_rating', 'guest_satisfaction_overall', 'weekend') are used as features
# X = df[['cleanliness_rating', 'guest_satisfaction_overall', 'weekend']]
# y = df['realSum']

# only the basic features ('host_is_superhost', 'room_private', 'person_capacity', 'bedrooms', 'entire_home_apt') are used as features
#X = df[['host_is_superhost', 'room_private', 'person_capacity', 'bedrooms', 'entire_home_apt']]
#y = df['realSum']

# only the spatial and basic features are used as features
#X = df[['lat', 'lng', 'dist', 'metro_dist', 'rest_index', 'attr_index', 'host_is_superhost', 'room_private', 'person_capacity', 'bedrooms', 'entire_home_apt']]
#y = df['realSum']

# only the quality and basic features are used as features
#X = df[['cleanliness_rating', 'guest_satisfaction_overall', 'weekend', 'host_is_superhost', 'room_private', 'person_capacity', 'bedrooms', 'entire_home_apt']]
#y = df['realSum']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

difference = []

best_model = None
best_params = None

def anlayse_predictions(y_test, prediction):
    # save the absolute difference between the predicted and the actual price for each row in the test set in a list
    for i in range(len(prediction)):
        difference.append(abs(prediction[i] - y_test.iloc[i]))
    # calculate the mean of the absolute differences
    mean = sum(difference) / len(difference)
    print("Mean of the absolute differences: ", mean)

def tree_regressor_with_hp_and_cv(X_train, y_train, X_test):
    global best_model
    global best_params

    # define the model
    reg = tree.DecisionTreeRegressor()

    # The hyperparameter tuning is restricted to following three parameters because they have the most significant impact
    # on the model's performance, and it would take too long to try out all possible combinations of hyperparameters.

    # max_depth: This parameter controls the maximum depth of the decision tree. A higher value can result in a more complex model that may overfit the training data, while a lower value can lead to underfitting. It's often a good starting point for hyperparameter tuning.
    # min_samples_split: This parameter specifies the minimum number of samples required to split an internal node. Increasing this value can prevent the tree from making overly specific decisions based on a small number of samples, reducing overfitting.
    # min_samples_leaf: This parameter defines the minimum number of samples required to be at a leaf node. Similar to min_samples_split, increasing this value can help prevent overfitting by ensuring a minimum number of samples in each leaf.

    # define the parameters to be tested (including default parameters)
    # define the parameters to search
    max_depth = [2, 5, 10, 50]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    # create a dictionary of all the parameter options
    # note has you can access the parameters of steps of a pipeline by using '__’
    parameters = dict(max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf)

    # define the grid search
    grid_search = GridSearchCV(estimator=reg, param_grid=parameters, cv=5, n_jobs=-1, verbose=1)

    # fit the grid search
    grid_search.fit(X_train, y_train)

    # save the best model and the best parameters for future use
    best_model = grid_search.best_estimator_

    # save the best parameters for future use
    best_params = grid_search.best_params_

    # use the best model to make predictions
    prediction = best_model.predict(X_test)

    # print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(grid_search.best_params_))
    print("Best score is {}".format(grid_search.best_score_))
    #anlayse_predictions(y_test, prediction)

    return best_model

def GetBestModel():
    return best_model

def GetBestParams():
    return best_params

#tree_regressor_with_hp_and_cv(X_train, y_train, X_test)
#print("best model: ", GetBestModel())
#print("best params: ", GetBestParams())

def getBestModel(featureGroup):

    # switch case for feature groups
    if featureGroup == 'all':
        '''
        best decision tree regressor with all features:
        
        #1    
        Best score is 0.4439168302114428
        Mean of the absolute differences:  51.3650477632412
        best model:  DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10)
        best params:  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10}
        
        #2
        Best score is 0.4435515487493025
        Mean of the absolute differences:  51.0894036929586
        best model:  DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10)
        best params:  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10}
        
        #3
        Best score is 0.44422234154380574
        Mean of the absolute differences:  48.86903723160596
        best model:  DecisionTreeRegressor(min_samples_leaf=2, min_samples_split=10)
        best params:  {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10}
        
        #4
        Best score is 0.44850392950278606
        Mean of the absolute differences:  51.30343001747159
        best model:  DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10)
        best params:  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10}
        
        #5
        Best score is 0.4366973983166364
        Mean of the absolute differences:  49.06380036744404
        best model:  DecisionTreeRegressor(max_depth=50, min_samples_leaf=2, min_samples_split=10)
        best params:  {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 10}
        
        The best model is the one with the highest score and the lowest mean of the absolute differences:
        DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10)
        '''
        return tree.DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10)

    elif featureGroup == 'spatial':

        '''
        best decision tree regressor with spatial features:

        #1
        Best score is 0.08824797614625882
        Mean of the absolute differences:  70.18184756777428
        best model:  DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=5)
        best params:  {'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 5}
        
        #2
        Best score is 0.08824797614625882
        Mean of the absolute differences:  70.18184756777427
        best model:  DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=5)
        best params:  {'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 5}
        
        #3
        Best score is 0.08824797614625879
        Mean of the absolute differences:  70.18184756777426
        best model:  DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=5)
        best params:  {'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 5}
        
        #4
        Best score is 0.08824797614625879
        Mean of the absolute differences:  70.18184756777428
        best model:  DecisionTreeRegressor(max_depth=2, min_samples_split=10)
        best params:  {'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 10}
        
        #5
        Best score is 0.08824797614625879
        Mean of the absolute differences:  70.18184756777426
        best model:  DecisionTreeRegressor(max_depth=2)
        best params:  {'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        The best model is the one with the highest score and the lowest mean of the absolute differences:
        DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=5)
        '''

        return tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=5)

    elif featureGroup == 'quality':
        '''
        best decision tree regressor with quality features:
        
        #1
        Best score is 0.030416188235722584
        Mean of the absolute differences:  73.94490572636442
        best model:  DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5}
        
        #2
        Best score is 0.03041618823572261
        Mean of the absolute differences:  73.94490572636444
        best model:  DecisionTreeRegressor(max_depth=5, min_samples_leaf=2)
        best params:  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}
        
        #3
        Best score is 0.030416188235722563
        Mean of the absolute differences:  73.94490572636442
        best model:  DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5}
        
        #4
        Best score is 0.03041618823572263
        Mean of the absolute differences:  73.94490572636442
        best model:  DecisionTreeRegressor(max_depth=5, min_samples_leaf=2)
        best params:  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}
        
        #5
        Best score is 0.03041618823572261
        Mean of the absolute differences:  73.94490572636444
        best model:  DecisionTreeRegressor(max_depth=5, min_samples_leaf=2)
        best params:  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}
        
        The best model is the one with the highest score and the lowest mean of the absolute differences:
        DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=5)
        '''

        return tree.DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=5)

    elif featureGroup == 'basic':
        '''
        best decision tree regressor with basic features:
        
        #1
        Best score is 0.30149906891513245
        Mean of the absolute differences:  57.96706045046162
        best model:  DecisionTreeRegressor(max_depth=10)
        best params:  {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #2
        Best score is 0.3013705965546728
        Mean of the absolute differences:  58.086728101171566
        best model:  DecisionTreeRegressor(min_samples_split=5)
        best params:  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5}

        #3
        Best score is 0.30149906891513245
        Mean of the absolute differences:  57.9670604504616
        best model:  DecisionTreeRegressor(max_depth=50)
        best params:  {'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #4
        Best score is 0.3013705965546728
        Mean of the absolute differences:  58.086728101171566
        best model:  DecisionTreeRegressor(max_depth=10, min_samples_split=5)
        best params:  {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}
        
        #5
        Best score is 0.3014990689151325
        Mean of the absolute differences:  57.967060450461595
        best model:  DecisionTreeRegressor(max_depth=10)
        best params:  {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        The best model is the one with the highest score and the lowest mean of the absolute differences:
        DecisionTreeRegressor(max_depth=10)
        '''

        return tree.DecisionTreeRegressor(max_depth=10)

    elif featureGroup == 'spatial_and_basic':
        '''
        best decision tree regressor with spatial and basic features:
        
        #1
        Best score is 0.40399243571464727
        Mean of the absolute differences:  40.536244207333084
        best model:  DecisionTreeRegressor()
        best params:  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #2
        Best score is 0.4054949829736649
        Mean of the absolute differences:  43.202442490881474
        best model:  DecisionTreeRegressor(max_depth=50, min_samples_leaf=2)
        best params:  {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 2}
        
        #3
        Best score is 0.4028129564649717
        Mean of the absolute differences:  44.680719975655045
        best model:  DecisionTreeRegressor(min_samples_leaf=2, min_samples_split=5)
        best params:  {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5}
        
        #4
        Best score is 0.40814423729364135
        Mean of the absolute differences:  39.6003856898973
        best model:  DecisionTreeRegressor(max_depth=50)
        best params:  {'max_depth': 50, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #5
        Best score is 0.3993991008179032
        Mean of the absolute differences:  48.763195999701615
        best model:  DecisionTreeRegressor(max_depth=50, min_samples_leaf=2, min_samples_split=10)
        best params:  {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 10}
        
        The best model is the one with the highest score and the lowest mean of the absolute differences:
        DecisionTreeRegressor(max_depth=50)
        '''

        return tree.DecisionTreeRegressor(max_depth=50)

    elif featureGroup == 'quality_and_basic':
        '''
        best decision tree regressor with quality and basic features:
        
        #1
        Best score is 0.32421313147277664
        Mean of the absolute differences:  57.21367144928647
        best model:  DecisionTreeRegressor(max_depth=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #2
        Best score is 0.3242131314727766
        Mean of the absolute differences:  57.21367144928648
        best model:  DecisionTreeRegressor(max_depth=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #3
        Best score is 0.32365996468955566
        Mean of the absolute differences:  57.21367144928649
        best model:  DecisionTreeRegressor(max_depth=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #4
        Best score is 0.32356603068791806
        Mean of the absolute differences:  57.21367144928651
        best model:  DecisionTreeRegressor(max_depth=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
        
        #5
        Best score is 0.3235660306879181
        Mean of the absolute differences:  57.21872063745648
        best model:  DecisionTreeRegressor(max_depth=5)
        best params:  {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}

        The best model is the one with the highest score and the lowest mean of the absolute differences:
        DecisionTreeRegressor(max_depth=5)
        '''

        return tree.DecisionTreeRegressor(max_depth=5)

    else:
        return None

'''
A

'''



