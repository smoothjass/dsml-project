import acquisition.acquisition as acq
import exploration.exploration as exp
import preprocessing.preprocessing as pre
import model.kNN as kNN

from sklearn.model_selection import train_test_split

# get the data from the .csv files (combines weekend and weekday data)
vienna = acq.get_vienna_data()

# explore the data (plots etc.) and clean it
# (remove outliers, check class imbalance, check encoding, dimensionality reduction of unnecessary columns)
vienna = exp.explore_and_clean(vienna)

# preprocess the data (normalize, scale)
vienna = pre.preprocess_data(vienna)

# now we can use the data in the different models
