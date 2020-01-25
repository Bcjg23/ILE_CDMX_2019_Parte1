#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os as os


def load_df(name_data, path_to_data = os.getcwd(), separator = ','):
    """
    This function loads the data file
    
    param:  name_data is a string with the name of the file we want to load
            path_to_data is a string with the path to where name_data is. Default is the current working directory
            sep is a string to indicate the separator. Default is ','
    return: data_frame with the information in the file name_data
    """
    #return pd.read_csv(path_to_data + '/' + name_data, sep = separator, parse_dates=True, dayfirst=True)
    return pd.read_csv(path_to_data + '/' + name_data, sep = separator)