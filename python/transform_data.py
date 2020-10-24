#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import clean_data


def split_string_column_into_numeric_columns(df, column_name, separator, new_cols_name):
    """
    This function splits the value in a column data frame that has string values into a separate numeric columns 
    
    params: df_name is a dataframe that contains the column we want to split
            colum_name is a string with the name of the column we want to split
            separtor is a string that contains the separator
            new_cols_name is a string list that contains the names for the splitted cols
    
    returns: df with the original column and the new columns named as in new_cols_name
    """
    #Split the column
    df[new_cols_name] =  df.loc[:, column_name ].str.split(separator, expand = True)
    
    for name in new_cols_name:
        df[name] = pd.to_numeric(df[name])
    
    return df


def split_string_column_into_numeric_columns(df, column_name, separator, new_cols_name):
    """This function split string column into numeric columns, given a separtor
    params:
       df: df where the columns exits
       column_name: str column that one wants to split
       separator: str indicating the token to use as separator (e.g. '-', '/')
       new_cols_name: list of str with the names of the new columsn
    return:"""
    #Split the column
    df[new_cols_name] =  df.loc[:, column_name ].str.split(separator, expand = True)
    for name in new_cols_name:
        df[name] = pd.to_numeric(df[name])
    return df

def clean_variables_values(df, columns_names):
    for name in columns_names:
        df[name]=pd.Series(estandariza_datos(pd.Index(df[name])))
    return df

def identify_date_variable(df):
    """This function identify date variables"""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                print(col)
            except ValueError:
                pass
    return df



def remove_whitespace(x):
    """ This function removes white spaces, if exists """
    try:
        x=" ".join(x.split())
    except:
        pass
    return x

def estandariza_datos(data): 
    """This function standardize the values of a series
    params:
        data: a pd.Series
    return :
        data: a pd.Series with values standardized"""
    data=pd.DataFrame(data)
    data=data.iloc[:,0]
    trans_table=str.maketrans('|°!"#$%&()=¿?¡}´]*¨{[-.,_:;<>','                             ')
    data=data.str.translate(trans_table)
    data=pd.DataFrame(data)
    data=data.applymap(remove_whitespace)
    data=data.iloc[:,0]
    data=data.str.lower().str.replace(' ','_')
    data=data.str.replace('ñ','n')
    trans_table=str.maketrans('áéíóúäëïöü','aeiouaeiou')
    data=data.str.translate(trans_table) 
    return data

def month_string_to_number_string(month_string):
    """This function changes month 3-letter abreviation to corresponding number of month.
    params:
        month_string: 3-letter abreviation in lower case of the month
    return:
        out: number of the month"""
    m = {
        'ene': '01',
        'feb': '02',
        'mar': '03',
        'abr': '04',
         'may': '05',
         'jun': '06',
         'jul': '07',
         'ago': '08',
         'sep': '09',
         'oct':'10',
         'nov':'11',
         'dic':'12'
        }
    s = month_string.strip()[:3].lower()

    try:
        out = m[s]
        return out
    except:
        raise ValueError('Not a month')
