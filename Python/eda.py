#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


def df_variables_info(df):
    """
    This function gives information about the df: number of observations, variables, type and number of variables
    
    params: df is a dataframe from which we want the information
    return: NONE???!!!!
    """
    # Number of variables
    print('Number of variables:', df.shape[1])
    
    #  Number of observations
    print('Number of observations:', df.shape[0])
    
    # Summary of the type of variables in the dataframe
    print('Type of variables in the dataframe: \n', df.dtypes.value_counts(),'\n')
    
    return None

def info_type_of_vars(df):
    """
    This function gives information about the type of variables in the df
    
    params: df dataframe
    return: list
    """
    # Identify numeric variables in the df  
    numeric_variables = df.select_dtypes(include = 'number').columns.values 
        
    # Identify categorical variables in the df  
    categorical_variables = df.select_dtypes(include = 'category').columns.values   
    
    # Identify date/time variables in the df  
    dates_variables = df.select_dtypes(include = 'datetime').columns.values   
        
    # Identify string variables in the df  
    string_variables = df.select_dtypes(include = 'object').columns.values   
        
    return numeric_variables, categorical_variables, dates_variables, string_variables

def print_info_type_of_vars(numeric_variables, categorical_variables, dates_variables, string_variables):
    print('Numericas:\n', numeric_variables,'\n', 'Total: ',len(numeric_variables),'\n')
    print('Categoricas:\n', categorical_variables,'\n', 'Total: ',len(categorical_variables),'\n')
    print('Formato Fecha: \n', dates_variables,'\n', 'Total: ', len(dates_variables), '\n')
    print('Foramto texto: \n', string_variables,'\n', 'Total: ', len(string_variables), '\n')
    return None

def data_profiling_num_vars(df, col_name):
    """
    This function computes basic statistics for the numerical column col_name in the dataframe df
    
    params: df a dataframe that contains the column col_name
            col_name a string with the name of the numerical variable from which we want to compute the statistics
    returns: dic a dictionary with the name and values of the basic statistics
    """
    top = df[col_name].value_counts().head(3).index.tolist()
    #Check all lists have the same lenght
    if len(top) != 3:
        if len(top) == 2:
            top.extend(['-'])
        else:
            top.extend(['-', '-'])
    
    dic = {'n_observations': df[col_name].count(),
           'max': df[col_name].max(),
           'min': df[col_name].min(),
           'mean': df[col_name].mean(),
           'std_dev': df[col_name].std(),
           '25%': df[col_name].quantile(0.25),
           'median': df[col_name].median(),
           '75%': df[col_name].quantile(0.75),
           'kurtosis': df[col_name].kurtosis(),
           'skewness': df[col_name].skew(),
           'n_unique_values': df[col_name].nunique(),
           '%_missings': df[col_name].isna().mean()*100,
           'Top1_most_common': top[0],
           'Top2_most_common': top[1],
           'Top3_most_common': top[2]
          }
           
    return dic




def descriptive_stats_for_numeric_vars(df, num_vars):
    """
    Show basic descriptive statistics for the numerical variables in the dataframe df
    
    param: df is a dataframe with numerical variables
           num_vars is a list with the names of the numerical variables in the df
           
    return: stat_table is a dataframe with the statistics for each numerical variable in df
    """
    stats_table = pd.DataFrame([data_profiling_num_vars(df, cols) for cols in num_vars]).transpose()
    stats_table.columns = num_vars
    
    return stats_table




def data_profiling_categorical_vars(df, col_name):
    """
    Show basic descriptive statistics for othe categorical variable col_name in the dataframe df
    
    param: df is a dataframe with the column col_name
           col_name is a string with the name of the categorical variable in the df
           
    return: dic a dictionary with name and value of the statistics computed
    """
    top = df[col_name].value_counts().head(3).index.tolist()
    #Check all lists have the same lenght
    if len(top) != 3:
        if len(top) == 2:
            top.extend(['-'])
        else:
            top.extend(['-', '-'])
    
    dic = {'n_observations': df[col_name].count(),
           'mode': df[col_name].mode()[0],
           'num_categories': len(df[col_name].unique()),
           'categories': df[col_name].unique(),
           'n_unique_values': df[col_name].nunique(),
           'n_missings': df[col_name].isna().sum(),
           '%_missings': df[col_name].isna().mean()*100,
           'Top1_most_common': top[0],
           'Top2_most_common': top[1],
           'Top3_most_common': top[2]
          }
    return dic
    




def descriptive_stats_for_categorical_vars(df, cat_vars):
    """
    Show basic descriptive statistics for the categorical variables in the dataframe df
    
    param: df is a dataframe with categorical variables
           cat_vars is a list of the categorical variables in the df
           
    return: cat_table is a dataframe with the statistics for each categorical variable in df
    """
    
    stats_table = pd.DataFrame([data_profiling_categorical_vars(df, cols) for cols in cat_vars]).transpose()
    stats_table.columns = cat_vars
    
    return stats_table


