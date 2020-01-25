#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer



def transforma_cat_dummies(df,cat_vars, drop_first_cat):
    """
    This function makes the One hot encoding for categorical variables
    
    params: df a dataframe with the categorical variables
            cat_vars a list with the name of the categorical variables to process
            drop_first_cat a boolean to indicate whether to get k-1 dummies out of k categorical levels by removing the first level
    return: df a dataframe with the encoded categorical variables
    """
    df = pd.get_dummies(df[cat_vars], columns = cat_vars, drop_first = drop_first_cat)
    
    return df


 
def transforma_num_a_binaria(df,var,nombre_nueva_var,percentil):
    """
    This function transform the variable we want to predict into a 2 categorical variables (in order to use chi-square test) according to a percentil (define 1 when de value of the variable is greater than that percentil)
    
    params: A dataframe that contain the variables we will use to predict (all binari variables)
    
    return: a binary variable
    """
    cuantil = df[[var]].quantile(percentil)[0]
    df = df.assign(nueva_binaria = lambda dataframe: dataframe[var].map(lambda nueva_binaria: 1 if nueva_binaria >= cuantil else 0))
    df = df[['nueva_binaria']]
    df.columns = [nombre_nueva_var]
    df = pd.DataFrame(df)
    
    return df



def chi_squared(var_cat_predictivas, var_cat_objetivo):
    """
    function that use chi-squared test
    
    params: categorical (binary) explicative variables , predictive variable
    
    return: results of the chi-squared test
    """
    aux=chi2(var_cat_predictivas,var_cat_objetivo)
    nombres=var_cat_predictivas.columns
    df = pd.DataFrame(columns=['var_cat_predictivas', 'chi_square_val', 'p_value'])
    df = df.astype( dtype={'var_cat_predictivas' : str, 'chi_square_val': float, 'p_value': float})
    n=len(aux[0])
    for i in range(n):
        df.loc[i] = [nombres[i], aux[0][i], aux[1][i]]
    df=df.assign(significativo = lambda dataframe: dataframe['p_value'].map(lambda significativo: 'significativo' if significativo >= 0.05 else 'no significativo'))
    
    
    return df



def mean_imputation_for_numeric_vars(df, num_vars):
    """
    This function imputes missing values with the mean
    
    param: df dataframe containing the numeric variables
           num_vars a list with the names of numeric variables
    
    return: vars_imputed a dataframe with the imputed values for each numerical variable with missings
    """
    aux = df[num_vars].values.reshape(df.shape[0],len(num_vars))
    vars_imputer = SimpleImputer(strategy="mean")
    vars_imputer.fit(aux)
    vars_imputed = vars_imputer.transform(aux)
    
    return vars_imputed


def standardization_for_numeric_vars(num_vars):
    """
    This function standardizes numerical variables
    
    param: num_vars a list with the names of numeric variables
    
    return: vars_standardized numeric variables standardized
    """
    ## Instanciando el estimador StandardScaler con los valores por default
    vars_standardization = StandardScaler()
    ## Aplicando el estimador a los datos
    vars_standardization.fit(num_vars)
    vars_standardized = vars_standardization.transform(num_vars)

    return vars_standardized




def backward_selection(y, x, pVal_threshold):
    """
    This function use the backward selection algoritm in order to select the most significant variables x to predict variable y
    
    param: x a dataframe thath contain the predict variables, y is the variable we want to predict, pVal_threshold indicate when the algoritm will stop
    
    return: the x variables that are significant to predict y
    """
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    p_vals = results.pvalues
    
    drop_index = p_vals.idxmax()

    while (p_vals[drop_index] > pVal_threshold):
        
        x = x.drop(drop_index, axis = 1)
        
        model = sm.OLS(y, x)
        results = model.fit()
        p_vals = results.pvalues
        
        
        drop_index = p_vals.idxmax()
        
    return x



