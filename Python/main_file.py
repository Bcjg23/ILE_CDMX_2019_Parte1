#!/usr/bin/env python
# coding: utf-8

#This is the main file to run the EDA for the data (consumo-agua.csv) for 2019 from https://datos.cdmx.gob.mx/explore/dataset/consumo-agua


### Change to the directory where your data file is:

# Update data file name
datafile_name = 'consumo-agua.csv'
# Update path to data file
path_to_datafile = '/home/bj/Documents/IntroDataScience/Tareas/eda_hw/Data'
# Path to save outputs
path_for_outputs = '/home/bj/Documents/IntroDataScience/Tareas/eda_hw/Results'


import load_data 
import clean_data
import transform_data
import eda
import geda



### 1. Load data 
consumo_agua = load_data.load_df(datafile_name, path_to_datafile, ';')

### 2. Standardization of name variables

# Initial information about the dataframe
eda.df_variables_info(consumo_agua)

# Identify numeric variables in the df  
numeric_variables = consumo_agua.select_dtypes(include = 'number').columns.values 
#print('Numeric variables:', numeric_variables)
    
# Identify categorical variables in the df  
categorical_variables = consumo_agua.select_dtypes(include = 'category').columns.values   
#print('Categorical variables:', categorical_variables)
    
# Identify date/time variables in the df  
dates_variables = consumo_agua.select_dtypes(include = 'datetime').columns.values   
#print('Date/Time variables:', dates_variables)

# Identify string variables in the df  
string_variables = consumo_agua.select_dtypes(include = 'object').columns.values   
#print('String variables:', string_variables)    

#Standardization of name variables
clean_consumo_agua = clean_data.clean_variables_names(consumo_agua)

# Check the columns name
#print(clean_consumo_agua.columns)

### 3. Transform the data
# Split the variable  variables
clean_consumo_agua = transform_data.split_string_column_into_numeric_columns(clean_consumo_agua, "geo_point", ',', \
                                                              ['latitud', 'longitud'])

# Drop geo_point and geo_shape variables
clean_consumo_agua.drop(['geo_point', 'geo_shape', 'gid'], axis = 1, inplace = True)

# Standardization of value variables
final_consumo_agua = transform_data.clean_variables_values(clean_consumo_agua, ['nomgeo', 'alcaldia', 'colonia', 'indice_des'])

################################  Save df to file  ##########################################
#final_consumo_agua.to_csv(path_for_outputs + '/' + 'final_consumo_agua.csv')
##############################################################################################

#Check changes and update variable arrays
#eda.df_variables_info(consumo_agua)

numeric_variables = final_consumo_agua.select_dtypes(include = 'number').columns.values 
#print('Numeric variables:', numeric_variables)

string_variables = final_consumo_agua.select_dtypes(include = 'object').columns.values   
#print('String variables:', string_variables)    


### 4. EDA
eda.descriptive_stats_for_numeric_vars(final_consumo_agua, numeric_variables)

eda.descriptive_stats_for_categorical_vars(final_consumo_agua,string_variables)

### GEDA

# Missing values
#geda.plot_missing_values(final_consumo_agua)

# Histograms for numeric variables
#geda.plot_histogram(final_consumo_agua, numeric_variables)

# Barplots for categorial variables
#geda.barplot_categorical_vars(final_consumo_agua,['alcaldia', 'indice_des'])

# Barplot of num variable by category
#geda.barplot_by_category(final_consumo_agua, 'consumo_total', 'alcaldia')

# Barplot of num variable by category
#geda.barplot_by_category(final_consumo_agua, 'consumo_total', 'indice_des')

# Boxplot for consumo
#geda.boxplot_all( final_consumo_agua, ['consumo_total_mixto', 'consumo_total_dom', 'consumo_total_no_dom'])

# Boxplot for consumo promedio
#geda.boxplot_all(final_consumo_agua, ['consumo_prom_mixto', 'consumo_prom_no_dom', 'consumo_prom_dom'])

# Correlation matrix
#geda.plot_heatmap_corr(final_consumo_agua, numeric_variables)

# Bivariate plot 
#geda.plot_by_pairs_grid(final_consumo_agua, ['consumo_total_mixto', 'consumo_total_dom', 'consumo_total_no_dom'])





