from matplotlib import pyplot as plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def explore_and_clean(city_data):
    # look at the columns
    print(city_data.columns)

    # what is column 'Unnamed: 0'?
    print(city_data['Unnamed: 0'].head())

    # drop the column 'Unnamed: 0'
    city_data.drop('Unnamed: 0', axis=1, inplace=True)

    # there is both 'rest_index' and 'rest_index_norm', which is normalized from 0 to 100
    # the same is true for 'attr_index' and 'attr_index_norm'
    # we can drop the normalized columns since we will be normalizing the data later anyway
    city_data.drop('rest_index_norm', axis=1, inplace=True)
    city_data.drop('attr_index_norm', axis=1, inplace=True)

    # look for duplicate rows
    print("Number of duplicate rows: ", city_data.duplicated().sum())

    # there seems to be a strong class imbalance for the class 'room_shared'
    print(city_data['room_shared'].value_counts())  # False: 3521, True: 16

    # since there are only 16 rows with 'room_shared' = True, this is not a useful feature
    # However, the rows with 'room_shared' = True are a lot cheaper, so they would bias the model
    # Therefore, we drop all rows with 'room_shared' = True
    city_data = city_data[city_data['room_shared'] == False]

    # now we also drop the column 'room_shared' since it is not useful anymore
    city_data = city_data.drop('room_shared', axis=1)

    # look at the data types
    print(city_data.dtypes)

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
    city_data.plot(kind="box", subplots=True, layout=(4, 3), figsize=(30, 30))
    plot.show()

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

    # print boxplot again to see how outliers have changed
    city_data.plot(kind="box", subplots=True, layout=(4, 3), figsize=(30, 30))
    plot.show()

    # other columns also have values that are more than 3 standard deviations away from the mean
    # however, these might significantly affect the price (like e.g. number of bedrooms)
    # so we will keep them for now

    # let's do some more data exploration
    # plot of the distribution of the target variable 'realSum' (price)
    plt.hist(city_data['realSum'], bins=100)
    plt.xlabel('Price')
    plt.ylabel('Number of listings')
    plt.title("Distribution of price")
    plt.show()

    # plot showing that entire homes/apartments are more common and more expensive
    plt.hist(city_data[city_data['entire_home_apt'] == True]['realSum'], bins=100, alpha=0.7, label='entire_home_apt')
    plt.hist(city_data[city_data['entire_home_apt'] == False]['realSum'], bins=100, alpha=0.7, label='private_room')
    plt.xlabel('Price')
    plt.ylabel('Number of listings')
    plt.title("Entire homes/apartments are more common and more expensive")
    plt.axvline(city_data[city_data['entire_home_apt'] == True]['realSum'].mean(), color='blue')
    plt.axvline(city_data[city_data['entire_home_apt'] == False]['realSum'].mean(), color='orange')
    plt.text(city_data[city_data['entire_home_apt'] == True]['realSum'].mean() + 15, 130,
             'Average: ' + str(round(city_data[city_data['entire_home_apt'] == True]['realSum'].mean(), 2)),
             color='blue', fontsize=14)
    plt.text(city_data[city_data['entire_home_apt'] == False]['realSum'].mean() + 15, 90,
             'Average: ' + str(round(city_data[city_data['entire_home_apt'] == False]['realSum'].mean(), 2)),
             color='darkorange', fontsize=14)
    plt.legend(labels=['Entire home/apt', 'Private room'])
    plt.show()

    # plot showing that listings with more bedrooms are more expensive
    plt.hist(city_data[city_data['bedrooms'] == 1]['realSum'], bins=100, alpha=0.7, label='1 bedroom')
    plt.hist(city_data[city_data['bedrooms'] == 2]['realSum'], bins=100, alpha=0.7, label='2 bedrooms')
    plt.hist(city_data[city_data['bedrooms'] == 3]['realSum'], bins=100, alpha=0.7, label='3 bedrooms')
    plt.hist(city_data[city_data['bedrooms'] == 4]['realSum'], bins=100, alpha=0.7, label='4 bedrooms')
    plt.xlabel('Price')
    plt.ylabel('Number of listings')
    plt.title("Listings with more bedrooms are more expensive")
    plt.axvline(city_data[city_data['bedrooms'] == 1]['realSum'].mean(), color='blue')
    plt.axvline(city_data[city_data['bedrooms'] == 2]['realSum'].mean(), color='orange')
    plt.axvline(city_data[city_data['bedrooms'] == 3]['realSum'].mean(), color='green')
    plt.axvline(city_data[city_data['bedrooms'] == 4]['realSum'].mean(), color='red')
    plt.text(city_data[city_data['bedrooms'] == 1]['realSum'].mean() + 15, 120,
             'Average: ' + str(round(city_data[city_data['bedrooms'] == 1]['realSum'].mean(), 2)),
             color='blue', fontsize=14)
    plt.text(city_data[city_data['bedrooms'] == 2]['realSum'].mean() + 15, 100,
             'Average: ' + str(round(city_data[city_data['bedrooms'] == 2]['realSum'].mean(), 2)),
             color='darkorange', fontsize=14)
    plt.text(city_data[city_data['bedrooms'] == 3]['realSum'].mean() + 15, 80,
             'Average: ' + str(round(city_data[city_data['bedrooms'] == 3]['realSum'].mean(), 2)),
             color='green', fontsize=14)
    plt.text(city_data[city_data['bedrooms'] == 4]['realSum'].mean() + 15, 60,
             'Average: ' + str(round(city_data[city_data['bedrooms'] == 4]['realSum'].mean(), 2)),
             color='red', fontsize=14)
    plt.legend(labels=['1 bedroom', '2 bedrooms', '3 bedrooms', '4 bedrooms'])
    plt.show()

    # however the price per bedroom is lower for listings with more bedrooms
    plt.bar(['1 bedroom', '2 bedrooms', '3 bedrooms', '4 bedrooms'],
            [city_data[city_data['bedrooms'] == 1]['realSum'].mean(),
             city_data[city_data['bedrooms'] == 2]['realSum'].mean() / 2,
             city_data[city_data['bedrooms'] == 3]['realSum'].mean() / 3,
             city_data[city_data['bedrooms'] == 4]['realSum'].mean() / 4])
    plt.ylabel('Average price per bedroom')
    plt.title("Price per bedroom is lower for listings with more bedrooms")
    plt.show()

    # most listings are for 2 or 4 people
    plt.bar(['1 person', '2 people', '3 people', '4 people', '5 people', '6 people'],
            [city_data[city_data['person_capacity'] == 1]['realSum'].count(),
             city_data[city_data['person_capacity'] == 2]['realSum'].count(),
             city_data[city_data['person_capacity'] == 3]['realSum'].count(),
             city_data[city_data['person_capacity'] == 4]['realSum'].count(),
             city_data[city_data['person_capacity'] == 5]['realSum'].count(),
             city_data[city_data['person_capacity'] == 6]['realSum'].count()])
    plt.ylabel('Number of listings')
    plt.title("Most listings are for 2 or 4 people")
    plt.show()

    # however the price per person is lower for listings with more people
    plt.bar(['2 people', '3 people', '4 people', '5 people', '6 people'],
            [city_data[city_data['person_capacity'] == 2]['realSum'].mean() / 2,
             city_data[city_data['person_capacity'] == 3]['realSum'].mean() / 3,
             city_data[city_data['person_capacity'] == 4]['realSum'].mean() / 4,
             city_data[city_data['person_capacity'] == 5]['realSum'].mean() / 5,
             city_data[city_data['person_capacity'] == 6]['realSum'].mean() / 6])
    plt.ylabel('Average price per person')
    plt.title("Price per person is lower for listings with more people")
    plt.show()

    # plot showing that price is only slightly higher on weekends
    plt.bar(['Weekday', 'Weekend'], [city_data[city_data['weekend'] == 0]['realSum'].mean(),
                                     city_data[city_data['weekend'] == 1]['realSum'].mean()])
    plt.ylabel('Average price')
    plt.title("Price is only slightly higher on weekends")
    plt.text(0.8, city_data[city_data['weekend'] == 0]['realSum'].mean() - 100,
             "+" + str(round((city_data[city_data['weekend'] == 1]['realSum'].mean() /
                              city_data[city_data['weekend'] == 0]['realSum'].mean() - 1) * 100, 2)) + '%',
             color='white', fontsize=16)
    plt.show()

    return city_data


def remove_outliers(standard_deviations, column_name, dataframe):
    # fit a normal distribution to the data
    # remove all rows that are more than the specified number of standard deviations away
    mean = dataframe[column_name].mean()
    std = dataframe[column_name].std()
    x = np.linspace(mean - standard_deviations * std, mean + standard_deviations * std, 100)
    plt.plot(x, norm.pdf(x, mean, std))

    # show mean and specified number of standard deviations
    plt.axvline(mean, color='red')
    plt.axvline(mean + standard_deviations * std, color='green')
    plt.axvline(mean - standard_deviations * std, color='green')

    # show the data
    plt.hist(dataframe[column_name], bins=100, density=True)

    # label the plot
    plt.xlabel(column_name)
    plt.ylabel('Probability density')
    plt.title("outliers in " + column_name)

    plt.show()

    # count the number of rows that are more than the specified number of standard deviations away from the mean
    print("Number of rows that are more than", standard_deviations, "standard deviations away from the mean in column",
          column_name, ":", len(dataframe[dataframe[column_name] > mean + standard_deviations * std]) + len(
            dataframe[dataframe[column_name] < mean - standard_deviations * std]))

    # remove these rows
    dataframe = dataframe[dataframe[column_name] < mean + standard_deviations * std]
    dataframe = dataframe[dataframe[column_name] > mean - standard_deviations * std]

    return dataframe
