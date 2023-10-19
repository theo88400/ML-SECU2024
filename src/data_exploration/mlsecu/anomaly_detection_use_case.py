import pandas as pd

from mlsecu.data_exploration_utils import get_column_names, get_nb_of_dimensions, get_unique_values, get_nb_of_rows, get_number_column_names, get_object_column_names
from mlsecu.data_preparation_utils import get_one_hot_encoded_dataframe, remove_nan_through_mean_imputation
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor



def get_list_of_attack_types(dataframe):
    """
    Retrieves the name of attack types of a pandas dataframe
    :param dataframe: input dataframe
    :return: the name of distinct attack types
    """
    if dataframe is None:
        return None
    return get_unique_values(dataframe, 'attack_type')

def get_nb_of_attack_types(dataframe):
    """
    Retrieves the number of distinct attack types of a pandas dataframe
    :param dataframe: input dataframe
    :return: the number of distinct attack types
    """
    if dataframe is None:
        return None
    return len(get_list_of_attack_types(dataframe))

def get_list_of_if_outliers(dataframe, outlier_fraction):
    """
    Extract the list of outliers according to Isolation Forest algorithm
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: list of outliers according to Isolation Forest algorithm
    """
    if dataframe is None:
        return None
    
    numbers = dataframe[get_number_column_names(dataframe)]
    objects = dataframe[get_object_column_names(dataframe)]
    number_dataframe = remove_nan_through_mean_imputation(numbers)
    object_dataframe = get_one_hot_encoded_dataframe(objects)

    dataframe = pd.concat([number_dataframe, object_dataframe], axis=1, sort=False)

    clf = IsolationForest(contamination=outlier_fraction, random_state=42)
    y_pred = clf.fit_predict(dataframe)
    return dataframe[y_pred == -1].index.values

def get_list_of_lof_outliers(dataframe, outlier_fraction):
    """
    Extract the list of outliers according to Local Outlier Factor algorithm
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: list of outliers according to Local Outlier Factor algorithm
    """
    if dataframe is None:
        return None
    dataframe = get_one_hot_encoded_dataframe(dataframe)
    dataframe = remove_nan_through_mean_imputation(dataframe)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
    y_pred = clf.fit_predict(dataframe)
    return dataframe[y_pred == -1].index.values

def get_list_of_parameters(dataframe):
    """
    Retrieves the list of parameters of a pandas dataframe
    :param dataframe: input dataframe
    :return: list of parameters
    """
    if dataframe is None:
        return None
    return get_column_names(dataframe).tolist()

def get_nb_of_if_outliers(dataframe, outlier_fraction):
    """
    Extract the number of outliers according to Isolation Forest algorithm
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: number of outliers according to Isolation Forest algorithm
    """
    outliers = get_list_of_if_outliers(dataframe, outlier_fraction)
    if outliers is None:
        return None
    return len(outliers)

def get_nb_of_lof_outliers(dataframe, outlier_fraction):
    """
    Extract the number of outliers according to Local Outlier Factor algorithm
    :param dataframe: input dataframe
    :param outlier_fraction: rate of outliers to be extracted
    :return: number of outliers according to Local Outlier Factor algorithm
    """
    outliers = get_list_of_lof_outliers(dataframe, outlier_fraction)
    if outliers is None:
        return None
    return len(outliers)

def get_nb_of_occurrences(dataframe):
    """
    Retrieves the number of occurrences of a pandas dataframe
    :param dataframe: input dataframe
    :return: number of occurrences
    """
    if dataframe is None:
        return None
    return get_nb_of_rows(dataframe)

def get_nb_of_parameters(dataframe):
    """
    Retrieves the number of parameters of a pandas dataframe
    :param dataframe: input dataframe
    :return: number of parameters
    """
    if dataframe is None:
        return None
    return get_nb_of_dimensions(dataframe)
