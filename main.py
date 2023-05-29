import acquisition.acquisition as acq
import exploration.exploration as exp
import preprocessing.preprocessing as pre
import model as model

# get the data from the .csv files (combines weekend and weekday data)
vienna = acq.get_vienna_data()

# explore the data (plots etc.) and clean it (remove outliers, check class imbalance etc.)
vienna = exp.explore_and_clean(vienna)

# preprocess the data (normalize, scale, encoding, dimensionality reduction etc.)
vienna = pre.preprocess_data(vienna)

# now we can use the data in the different models
