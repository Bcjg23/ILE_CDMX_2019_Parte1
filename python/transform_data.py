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

# def clean_variables_values(df, columns_names):
#     """
#     Clean the values of the selected columns: lowercase, no spaces, no accent marks
    
#     param: df is a dataframe where the variables to clean are
#            colum_names is a list with the name of the variables we are going to clean
#     return: df with clean colums
#     """
#     for name in columns_names:
#         df[name] = df[name].str.strip().str.lower().str.replace('.','').str.replace(',','').str.replace(';','').str.replace(' ','_').str.replace('á','a').str.replace('é','e')      .str.replace('í','i').str.replace('ó','o').str.replace('ú','u').str.replace('ñ','n') 
    
#     return df

def split_string_column_into_numeric_columns(df, column_name, separator, new_cols_name):
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
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                print(col)
            except ValueError:
                pass
    return df

# def estandariza_datos(data):  
#     data=pd.DataFrame(data)
#     data=data.applymap(clean_data.remove_whitespace)
#     data=data.iloc[:,0]
#     data=data.str.lower().str.replace(' ','_')
#     data=data.str.replace('ñ','n')
#     trans_table=str.maketrans('áéíóúäëïöü','aeiouaeiou')
#     data=data.str.translate(trans_table) 
#     return data

def remove_whitespace(x):
    try:
        x=" ".join(x.split())
    except:
        pass
    return x

def estandariza_datos(data):  
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
