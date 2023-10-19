import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# get_one_hot_encoded_dataframe(dataframe)
# Retrieves the one hot encoded dataframe
 
# :param dataframe: input dataframe
# :return: the associated one hot encoded dataframe

def get_one_hot_encoded_dataframe(dataframe):
    if dataframe is None:
        return None
    return pd.get_dummies(dataframe)


# remove_nan_through_mean_imputation(dataframe)
# Remove NaN (Not a Number) entries through mean imputation
 
# :param dataframe: input dataframe
# :return: the dataframe with  NaN (Not a Number) entries replaced using mean imputation

def remove_nan_through_mean_imputation(dataframe):
    if dataframe is None:
        return None
    return dataframe.fillna(dataframe.mean())
