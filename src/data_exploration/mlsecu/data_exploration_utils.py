import pandas as pd
import numpy as np


def get_column_names(dataframe):
    """
    Get the name of columns in the dataframe
    :param dataframe: input dataframe
    :return: name of columns
    """
    return dataframe.columns.values

def get_nb_of_dimensions(dataframe):
    """
    Get the number of dimensions
    :param dataframe: input dataframe
    :return: number of dimensions
    """
    return dataframe.shape[1]

def get_nb_of_rows(dataframe):
    """
    Get the number of rows
    :param dataframe: input dataframe
    :return: number of rows
    """
    if dataframe is None:
        return None
    return dataframe.shape[0]

def get_number_column_names(dataframe):
    """
    Get the number of object columns
    :param dataframe: input dataframe
    :return: number of object columns
    """
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include=['number']).columns.values.tolist()

def get_object_column_names(dataframe):
    """
    Get the name of object columns
    :param dataframe: input dataframe
    :return: name of object columns
    """
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include=['object']).columns.values.tolist()

def get_unique_values(dataframe, column_name):
    """
    Get the unique values for a given column
    :param dataframe: input dataframe
    :param column_name: target column label
    :return: unique values for a given column
    """
    if dataframe is None or column_name is None or column_name == '' or column_name not in dataframe.columns.values:
        return None
    return dataframe[column_name].unique()
