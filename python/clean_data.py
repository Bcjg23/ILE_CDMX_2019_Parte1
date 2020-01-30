#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np


# def clean_variables_names(data_frame):
#     """
#     Clean the name of the columns for the data_frame: lowercase, no spaces, no accent marks
    
#     param: data_frame to clean column names
#     return: data_frame with clean colum names
#     """
    
#     data_frame.columns = data_frame.columns.str.strip().str.lower().str.replace(' ','_')                  .str.replace('á','a').str.replace('é','e').str.replace('í','i')                      .str.replace('ó','o').str.replace('ú','u').str.replace('ñ','n')
#     return data_frame



def clean_variables_names(df):
    """
    Clean the name of the columns for the data_frame: lowercase, no spaces, no accent marks
    
    param: data_frame to clean column names
    return: data_frame with clean colum names
    """
    df.columns=estandariza_datos(df.columns)
    return df

def remove_whitespace(x):
    try:
        x=" ".join(x.split())
    except:
        pass
    return x

#def remove_especific_characters(x,chars):
#    aux="[" + chars + "]"
#    try:
#        x=x.str.re.sub(aux,"",x)
#     except:
#         print("chalesss")
#         pass
#     return x

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
