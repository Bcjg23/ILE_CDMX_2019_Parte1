#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def barplot_categorical_vars(df, cat_vars):
    """
    Show basic graphics for the categorical variables in the dataframe df
    
    param: df is a dataframe with categorical variables
           cat_vars is a list of the categorical variables in the df
           
    return: NONE 
    """
    for var in cat_vars:
        # Plot bar chart
        df[var].value_counts().plot(kind = 'barh', figsize = (8, 6))
        plt.xlabel("Frecuancia", labelpad = 14)
        plt.ylabel(var, labelpad = 14)
        titulo_graf = "Grafica frecuencia " + var
        plt.title(titulo_graf, y = 1.02)
        #plt.savefig(titulo_graf + ".png")
        plt.show()
    


def plot_missing_values(df):
    """
    This function shows the missing values for all the variables in df
    """
    sns.heatmap(df.isnull(), cbar=False)
    


def plot_histogram(df, num_vars):
    """
    This function plots the histogram fo each of the variables in num_vars
    
    params: df a dataframe that have numerical variables
            num_vars the list of the numerical variables in df
    returns: histograms
    """
    for var in num_vars:
        plt.figure(figsize=(8,5))
        hist = sns.distplot(df[var].dropna(), kde = False)
        titulo_graf = "Histograma " + var
        hist.set_title(titulo_graf)
        hist.set_ylabel("Frecuencia")
        #plt.savefig(titulo_graf + ".png")
        plt.show()



def boxplot_all(df, num_vars):
    """
    This function plots the boxplot for all the variables in num_vars in one graph
    
    params: df a dataframe that have numerical variables
            num_vars the list of the numerical variables in df
    returns: one graph with multiple boxplots
    """
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10,8))
    sns.boxplot(y="variable", x="value", data=pd.melt(df[num_vars].dropna()), palette = 'Blues')
    
    plt.show()
     



def plot_by_pairs_grid(df, num_vars):
    """
    This function plots the a grid of small subplots using scatter plots. Each row and column is assigned to a 
    different variable, so the resulting plot shows each pairwise relationship for all the variables in num_vars
    
    params: df a dataframe that have numerical variables
            num_vars the list of the numerical variables in df
    returns: one graph with multiple subplots
    """
    g = sns.PairGrid(df[num_vars].dropna())
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    


def plot_heatmap_corr(df, num_vars):
    """
    This function plots a Heatmap of correlation matrix for numerical variables. When plotting a correlation matrix
    any non-numeric column is ignores. Categorical variables can be changed to numeric variables.
    
    params: df a dataframe that have numerical variables
            num_vars the list of the numerical variables in df
    returns: Heatmap of correlation matrix for variables in num_vars
    """
    
    mask_matrix = np.triu(df[num_vars].corr())
    
    plt.figure(figsize=(15,10))
    
    sns.heatmap(df[num_vars].corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask = mask_matrix)

    
    
    
def plot_bivariate_hist(df, var1, var2):
    """
    This function plots a histogram for 2 numerical variables in the same graph
    """
    sns.distplot( df[var1] , color="skyblue", label="Sepal Length")
    sns.distplot( df[var2] , color="red", label="Sepal Width")
    
    
    
def barplot_by_category(df, var, cat):
    """
    This function shows a barplot for a numerical variable var by categories in cat
    """
    sns.set(style="whitegrid", palette = 'deep')
    
    plt.figure(figsize=(10,8))
        
    x_data = df.groupby(cat)[var].mean().sort_values(ascending = False)
    titulo_graf = "Promedio de " + str(var) + " por " + str(cat) 
    
    if (len(x_data)>20):
        x_data = x_data[0:19]
        titulo_graf = titulo_graf + " (Top 20)"
        
    x_plot = sns.barplot(x=x_data.values, y=x_data.index, data=df)
    x_plot.set_title(titulo_graf)
    plt.show()