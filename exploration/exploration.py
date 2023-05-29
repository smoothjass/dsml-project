from matplotlib import pyplot as plt


def explore_and_clean(city_data):

    # look at the columns
    print(city_data.columns)

    # what is column 'Unnamed: 0'?
    print(city_data['Unnamed: 0'].head())

    # drop the column 'Unnamed: 0'
    city_data.drop('Unnamed: 0', axis=1, inplace=True)

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
    city_data.boxplot()

    # print boxplot
    plt.show()

    return city_data