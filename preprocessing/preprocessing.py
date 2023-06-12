from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def preprocess_data(city_data):
    # we use a MinMaxScaler to scale the data to the range [0,1]
    # scaler = MinMaxScaler()
    # trying the standard scaler for faster convergence
    scaler = StandardScaler()

    # scale all columns except the target column 'realSum' and boolean columns
    columns_to_scale = ['person_capacity', 'cleanliness_rating', 'guest_satisfaction_overall', 'bedrooms', 'dist',
                        'metro_dist', 'attr_index', 'rest_index', 'lng', 'lat']
    city_data[columns_to_scale] = scaler.fit_transform(city_data[columns_to_scale])

    # standardize all columns except the target column 'realSum' and boolean columns
    # we use Z-transform (x - mean) / std
    columns_to_standardize = ['person_capacity', 'cleanliness_rating', 'guest_satisfaction_overall', 'bedrooms', 'dist',
                              'metro_dist', 'attr_index', 'rest_index', 'lng', 'lat']
    # city_data[columns_to_standardize] = (city_data[columns_to_standardize] - city_data[columns_to_standardize].mean()) / city_data[columns_to_standardize].std()

    return city_data
