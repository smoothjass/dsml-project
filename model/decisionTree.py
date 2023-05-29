from matplotlib import pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import preprocessing.preprocessing as pre

def get_vienna_data():
    vienna_weekdays = pd.read_csv('../datasets/vienna_weekdays.csv', sep=',')
    vienna_weekdays['weekend'] = False
    vienna_weekend = pd.read_csv('../datasets/vienna_weekends.csv', sep=',')
    vienna_weekend['weekend'] = True
    vienna = pd.concat([vienna_weekend, vienna_weekdays], ignore_index=True, sort=False)
    return vienna

def remove_outliers(standard_deviations, column_name, dataframe):
    # fit a normal distribution to the data
    # remove all rows that are more than the specified number of standard deviations away
    mean = dataframe[column_name].mean()
    std = dataframe[column_name].std()
    x = np.linspace(mean - standard_deviations * std, mean + standard_deviations * std, 100)
    plt.plot(x, norm.pdf(x, mean, std))

    # show mean and specified number of standard deviations
    #plt.axvline(mean, color='red')
    #plt.axvline(mean + standard_deviations * std, color='green')
    #plt.axvline(mean - standard_deviations * std, color='green')

    # show the data
    #plt.hist(dataframe[column_name], bins=100, density=True)

    # label the plot
    #plt.xlabel(column_name)
    #plt.ylabel('Probability density')

    #plt.show()

    # count the number of rows that are more than the specified number of standard deviations away from the mean
    #print("Number of rows that are more than {} standard deviations away from the mean: ".format(standard_deviations),
          #len(dataframe[dataframe[column_name] > mean + standard_deviations * std]) + len(
              #dataframe[dataframe[column_name] < mean - standard_deviations * std]))

    # remove these rows
    dataframe = dataframe[dataframe[column_name] < mean + standard_deviations * std]
    dataframe = dataframe[dataframe[column_name] > mean - standard_deviations * std]

    return dataframe


def explore_and_clean(city_data):
    # look at the columns
    # print(city_data.columns)

    # what is column 'Unnamed: 0'?
    # print(city_data['Unnamed: 0'].head())

    # drop the column 'Unnamed: 0'
    city_data.drop('Unnamed: 0', axis=1, inplace=True)

    # there is both 'rest_index' and 'rest_index_norm', which is normalized from 0 to 100
    # the same is true for 'attr_index' and 'attr_index_norm'
    # we can drop the normalized columns since we will be normalizing the data later anyway
    city_data.drop('rest_index_norm', axis=1, inplace=True)
    city_data.drop('attr_index_norm', axis=1, inplace=True)

    # look for duplicate rows
    # print("Number of duplicate rows: ", city_data.duplicated().sum())

    # there seems to be a strong class imbalance for the class 'room_shared'
    # print(city_data['room_shared'].value_counts())  # False: 3521, True: 16

    # since there are only 16 rows with 'room_shared' = True, this is not a useful feature
    # However, the rows with 'room_shared' = True are a lot cheaper, so they would bias the model
    # Therefore, we drop all rows with 'room_shared' = True
    city_data = city_data[city_data['room_shared'] == False]

    # now we also drop the column 'room_shared' since it is not useful anymore
    city_data = city_data.drop('room_shared', axis=1)

    # look at the data types
    # print(city_data.dtypes)

    # the column 'room_type' has strings but there are only 2 different values
    # there is only "Entire home/apt" and "Private room"
    # we can encode this column with a bool for 'entire_home_apt'
    city_data['entire_home_apt'] = city_data['room_type'] == 'Entire home/apt'

    # now we can drop the column 'room_type'
    city_data = city_data.drop('room_type', axis=1)

    # column 'multi' has 1 and 0 values, but is marked as int64 (should be bool)
    # we can convert it to bool
    city_data['multi'] = city_data['multi'].astype(bool)

    # same for 'biz' column
    city_data['biz'] = city_data['biz'].astype(bool)

    # create boxplots to find outliers
    #city_data.plot(kind="box", subplots=True, layout=(4, 3), figsize=(30, 30))
    #plot.show()

    # there are 3 obvious outliers in 'realSum'
    # the fourth-highest value is 892
    # and the three outliers are over 10,000
    # we drop these three rows
    city_data = city_data[city_data['realSum'] < 10000]

    # there are also obvious outliers in 'rest_index' and
    city_data = city_data[city_data['rest_index'] < 2000]

    # there are some outliers in 'guest_satisfaction_overall'
    # we probably want to remove them since we don't have enough data to predict for very low values here
    # we fit a normal distribution to the data and remove all rows that are more than 3 standard deviations away

    city_data = remove_outliers(3, 'guest_satisfaction_overall', city_data)

    # also remove outliers in 'cleanliness_rating', 'rest_index', 'metro_dist' and 'attr_index'
    city_data = remove_outliers(5, 'cleanliness_rating', city_data)
    city_data = remove_outliers(5, 'rest_index', city_data)
    city_data = remove_outliers(5, 'attr_index', city_data)
    city_data = remove_outliers(5, 'metro_dist', city_data)

    # other columns also have values that are more than 3 standard deviations away from the mean
    # however, these might significantly affect the price (like e.g. number of bedrooms)
    # so we will keep them for now

    # print boxplot again to see if there are still outliers
    #city_data.plot(kind="box", subplots=True, layout=(4, 3), figsize=(30, 30))
    #plot.show()

    return city_data

# get the data from the .csv files (combines weekend and weekday data)
vienna = get_vienna_data()

# explore the data (plots etc.) and clean it (remove outliers, check class imbalance etc.)
vienna = explore_and_clean(vienna)

# preprocess the data (normalize, scale, encoding, dimensionality reduction etc.)
vienna = pre.preprocess_data(vienna)

df = vienna.copy()
X = df.drop(columns=['realSum'])
y = df['realSum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

difference = []

def anlayse_predictions(y_test, prediction):
    # save the absolute difference between the predicted and the actual price for each row in the test set in a list
    for i in range(len(prediction)):
        difference.append(abs(prediction[i] - y_test.iloc[i]))
    # calculate the mean of the absolute differences
    mean = sum(difference) / len(difference)
    print("mean: ", mean)

# predict the price with default hyperparameters
def default_hp(X_train, y_train, X_test, y_test):
    reg = tree.DecisionTreeRegressor()
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    score = reg.score(X_test, y_test)
    print("#dt-1: default parameters - score: ", score)
    anlayse_predictions(y_test, prediction)


default_hp(X_train, y_train, X_test, y_test)


# Increasing the min_samples_split prevents the tree from splitting nodes that have a smaller number of samples, potentially reducing overfitting.
def min_samples_split(X_train, y_train, X_test, y_test, min_samples_split):
    reg = tree.DecisionTreeRegressor(min_samples_split=min_samples_split)
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-2: min_samples_split = ", min_samples_split, " - score: ", score)

min_samples_split(X_train, y_train, X_test, y_test, 2)

# The DecisionTreeRegressor uses the mean squared error as the default splitting criterion (mse).
# You can experiment with other criteria, such as friedman_mse or mae, to see if they result in better performance.

def firedman_mse(X_train, y_train, X_test, y_test):
    reg = tree.DecisionTreeRegressor(criterion='friedman_mse')
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-3: friedman_mse - score: ", score)

def absolute_error(X_train, y_train, X_test, y_test):
    reg = tree.DecisionTreeRegressor(criterion='absolute_error')
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-4: absolute_error - score: ", score)

firedman_mse(X_train, y_train, X_test, y_test)

# The max_depth parameter determines when the splitting up of the decision tree stops.
# The default setting is that the nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# Try setting max_depth to 3, 5, 10 and None and compare the scores.

def max_depth(X_train, y_train, X_test, y_test, max_depth):
    reg = tree.DecisionTreeRegressor(max_depth=max_depth)
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-5: max_depth = ", max_depth, " - score: ", score)

max_depth(X_train, y_train, X_test, y_test, 3)
max_depth(X_train, y_train, X_test, y_test, 5)
max_depth(X_train, y_train, X_test, y_test, 10)
max_depth(X_train, y_train, X_test, y_test, None)

# The min_samples_leaf parameter can also be tweaked to set the minimum number of samples required to be at a leaf node.
# Try setting min_samples_leaf to 1, 2, 5 and 10 and compare the scores.

def min_samples_leaf(X_train, y_train, X_test, y_test, min_samples_leaf):
    reg = tree.DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-6: min_samples_leaf = ", min_samples_leaf, " - score: ", score)

min_samples_leaf(X_train, y_train, X_test, y_test, 1)
min_samples_leaf(X_train, y_train, X_test, y_test, 2)
min_samples_leaf(X_train, y_train, X_test, y_test, 5)
min_samples_leaf(X_train, y_train, X_test, y_test, 10)

# The max_leaf_nodes parameter can also be tweaked to reduce the number of leaf nodes in the tree.
# Try setting max_leaf_nodes to 5, 10, 20 and 100 and compare the scores.

def max_leaf_nodes(X_train, y_train, X_test, y_test, max_leaf_nodes):
    reg = tree.DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-7: max_leaf_nodes = ", max_leaf_nodes, " - score: ", score)

max_leaf_nodes(X_train, y_train, X_test, y_test, 5)
max_leaf_nodes(X_train, y_train, X_test, y_test, 10)
max_leaf_nodes(X_train, y_train, X_test, y_test, 20)
max_leaf_nodes(X_train, y_train, X_test, y_test, 100)

# The min_impurity_decrease parameter can also be tweaked to set the minimum impurity decrease required to split a node.
# Try setting min_impurity_decrease to 0.0, 0.1, 0.2 and 0.5 and compare the scores.

def min_impurity_decrease(X_train, y_train, X_test, y_test, min_impurity_decrease):
    reg = tree.DecisionTreeRegressor(min_impurity_decrease=min_impurity_decrease)
    reg = reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    anlayse_predictions(y_test, prediction)
    score = reg.score(X_test, y_test)
    print("#dt-8: min_impurity_decrease = ", min_impurity_decrease, " - score: ", score)

min_impurity_decrease(X_train, y_train, X_test, y_test, 0.0)
min_impurity_decrease(X_train, y_train, X_test, y_test, 0.1)
min_impurity_decrease(X_train, y_train, X_test, y_test, 0.2)
min_impurity_decrease(X_train, y_train, X_test, y_test, 0.5)


