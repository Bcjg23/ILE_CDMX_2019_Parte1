#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas_profiling
import xgboost as xgb

from matplotlib.gridspec import GridSpec
from datetime import datetime, date, time, timedelta
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, MinMaxScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit #Para split temporal
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, auc, roc_curve

# Import created functions (in .py)
import load_data 
import clean_data
import transform_data
import eda
import geda
import feature_selection


# Introduction
# 
# The project objective is to predict if a women sekking legal abortion services in Mexico City is 23 years or older.
# The result of this project will help shape the target population of a campaign of medical services for women seeking
# abortion services. The campaign will offer low-cost or free salpingoclasia (permanent sterilization) services and encourage
# women to participate in other social welfare programs to which they may be eligible (for example, single mothers cash
# transfers).
#
# The dataset used is the legal abortion [Interrupciones Legales de Embarazo] dataset for Mexico City.
#
# Full project (Spanish): https://github.com/Bcjg23/ILE_CDMX_2019_Parte1
# Final report (Spanish): https://bcjg23.github.io/ILE_CDMX_2019_Parte1/ILE_Reporte_Final.html

##################################################  LOAD DATASET #############################################################
# Update data file name
datafile_name = 'interrupcion-legal-del-embarazo.csv'

# Update path to data file
path_to_datafile = '~/Documents/Brenda/Proyecto/Data'
#path_to_datafile = '/home/bj/Documents/IntroDataScience/Proyecto/Data'
#path_to_datafile ='C:/Users/GZARAZUA/Desktop/MCD/MINERIA DE DATOS/Reporte aborto'

# Load the dataset
interrupcion_legal = load_data.load_df(datafile_name, path_to_datafile, ';')


########################################  CLEAN NAME VARIABLES  ######################################################
# Columns names standardization
clean_interrupcion_legal=clean_data.clean_variables_names(interrupcion_legal)

######################################## DATASET GENERAL INFORMATION #################################################
# General information
eda.df_variables_info(clean_interrupcion_legal)
# Classify variables per type
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(clean_interrupcion_legal)
# Print to screen variables per type
eda.print_info_type_of_vars(numeric_variables, categorical_variables, dates_variables, string_variables)conf_mat.sum(1)

#######################################  CLEAN VARIABLES VALUES  ####################################################
# Standardization of the values of "string" variables. 
# Except fingreso, fmenstrua, h_fingres that are "date" variables, but stored as "string"
clean_interrupcion_legal = transform_data.clean_variables_values(
    clean_interrupcion_legal, ['mes','autoref','edocivil_descripcion','desc_derechohab','nivel_edu','ocupacion',
                               'religion','parentesco','entidad','alc_o_municipio','consejeria','anticonceptivo','motiles',
                               'desc_servicio','p_consent','procile','s_complica','panticoncep'])

#######################################  ADJUST TYPES OF VARIABLES ################################################
# Change type of "fmenstrua" to "date" type
clean_interrupcion_legal['fmenstrua'] = pd.to_datetime(clean_interrupcion_legal['fmenstrua'])

# Change type of "fingreso" to "date" type
# Change NA to np.nan
clean_interrupcion_legal['fingreso'] = clean_interrupcion_legal['fingreso'].replace(np.nan,"NA")
# Split fingreso in (year, month, day) and check for errors in dates
dividir = lambda x: pd.Series([i for i in x.split('/')])
aux = clean_interrupcion_legal['fingreso'].apply(dividir)
aux = pd.DataFrame(aux)
aux.columns = ['dia','mes','ano']
# Check for months > 12
mes_mayor_12 = aux[pd.to_numeric(aux['mes']) > 12]
# Check years before 2016 o greater than 2019 could be errors
ano_menor_16_mayor_19 = aux[(pd.to_numeric(aux['ano'])<16) | (pd.to_numeric(aux['ano'])>19)]
# Get index with errors
no_observ = mes_mayor_12.loc[mes_mayor_12['mes']=='0719'].index.values[0]
# Set rigth date
clean_interrupcion_legal['fingreso'].iloc[no_observ] = '17/07/19'


# Adjut variable 'año' in 'fingreso' according to variable 'ano' 
# Differences are assumed as errors
indx = ano_menor_16_mayor_19.index.values
clean_interrupcion_legal.loc[indx, 'fingreso'] = ano_menor_16_mayor_19.iloc[indx, 0] +'/' + ano_menor_16_mayor_19.iloc[indx, 1] + '/' + str(clean_interrupcion_legal.iloc[indx,'ano'])[-2:]

# After correcting errors, we change to "date"
clean_interrupcion_legal['fingreso'] = pd.to_datetime(clean_interrupcion_legal['fingreso'])

# Update classification of variables by type
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(clean_interrupcion_legal)
# eda.print_info_type_of_vars(numeric_variables, categorical_variables, dates_variables, string_variables)

#######################################  CORRECT ORTHOGRAPHIC ERRORS #########################################
# Variable: autoref
# Misspelling 'autorreferida' as 'autorefereida'
clean_interrupcion_legal.loc[clean_interrupcion_legal['autoref']=='autoreferida', 'autoref']='autorreferida'

# Variable: desc_derechohab...
clean_interrupcion_legal.loc[clean_interrupcion_legal['edocivil_descripcion']=='n/e', 'edocivil_descripcion']='ne'

# Variables: fmenstrua
# Values of 'fmenstrua' greater than 2019 are assumed as errors and changed to NaN
clean_interrupcion_legal.loc[clean_interrupcion_legal['fmenstrua'] > pd.Timestamp(date(2019,12,31)), 'fmenstrua']=pd.to_datetime(np.nan)

####################################### DATA TRASFORMATION ##################################################
# Create target variable
clean_interrupcion_legal['23_o_mayor'] = np.where(clean_interrupcion_legal['edad'] >= 23, 1, 0)

# Create temporal variable "ano_mes_ile"
mes_num = clean_interrupcion_legal['mes'].map(transform_data.month_string_to_number_string)
clean_interrupcion_legal['ano_mes_ile'] = clean_interrupcion_legal['ano'].astype(str) + "/" + mes_num.astype(str)
# Adjust type to "date"
clean_interrupcion_legal['ano_mes_ile'] = pd.to_datetime(clean_interrupcion_legal['ano_mes_ile'])

# Group age by quintils
clean_interrupcion_legal['edad_quintiles'] = pd.cut(clean_interrupcion_legal['edad'], 
                                                    bins=[0,4,9,14,19,24,29,34,39,44,49,54,59,100],
                                                    labels=["0 a 4", "5 a 9", "10 a 14", "15 a 19", "20 a 24", "25 a 29",
                                                            "30 a 34", "35 a 39", "40 a 44", "45 a 49", "50 a 54", "55 a 59",
                                                            "60 y mas"], right=True)

# Recategorize civil state  variable 'estado_civil'
dic_edo = {'soltera':'soltera', 'divorciada':'divorciada', 'union_libre':'union_libre', 'casada':'casada',
           'separada':'separada', 'n/e': 'no_especificado', 'na':'no_especificado'}
clean_interrupcion_legal['edo_civil'] = clean_interrupcion_legal['edocivil_descripcion'].map(dic_edo)

# Recategorize education level variable ('nivel_educativo')
dic_escol = {'ninguno':'ninguno', 'preescolar':'ninguno', 'primaria_completa':'primaria', 'primaria_incompleta':'primaria',
             'secundaria_incompleta':'secundaria', 'secundaria_completa':'secundaria',     
             'preparatoria_incompleta':'media_superior',
             'preparatoria_completa':'media_superior', 'carrera_tecnica':'superior', 
             'licenciatura_incompleta':'superior','licenciatura_completa':'superior',
             'maestria':'posgrado', 'doctorado':'posgrado', 'posgrado':'posgrado', 'posgrado_incompleto':'posgrado',
             'no_especifica':'no_especificado', 'NA':'no_especificado', 'otra':'otra'}
clean_interrupcion_legal['escolaridad'] = clean_interrupcion_legal['nivel_edu'].map(dic_escol)

# Recategorize ocupation variable ('ocupacion')
dic_ocup = {'abogada':'profesionista', 'administradora':'profesionista', 'ama_de_casa':'ama_de_casa',
            'arquitecta':'profesionista', 'artista':'profesionista', 'asesora_financiera':'profesionista',
            'asesora_juridica':'profesionista', 'auditora':'profesionista',
            'bibliotecaria':'profesionista', 'cajera':'empleada', 'capturista':'empleada', 'chofer':'empleada',
            'cientifica':'profesionista', 'cocinera':'empleada', 'comerciante':'empleada', 
            'constructora_o_elecetricista':'empleada', 'constructura_o_elecetricista':'empleada',
            'contadora':'empleada', 'dentista':'profesionista', 'desempleada':'desempleada', 'diseñadora':'profesionista',
            'ejecutiva':'profesionista', 'empleada':'empleada', 'enfermera':'profesionista', 'estilista':'empleada',
            'estudiante':'estudiante', 'fisioterapeuta':'empleada', 'fotografa':'empleada', 
            'informatica_o_tecnologia':'profesionista', 'ingeniera':'profesionista', 'logistica_o_eventos':'empleada', 
            'mecanica':'profesionista', 'medico':'profesionista', 'mesera':'empleada', 'modelo':'empleada', 
            'na':'no_especificado', 'nutiologa':'profesionista', 'obrera':'empleada',
            'paramedico':'empleada', 'periodista_o_publicista':'profesionista', 'policia_o_seguridad':'empleada',
            'profesora_o_educadora':'profesionista', 'psicologa':'profesionista', 'recepcionista':'empleada',
            'recursos_humanos':'empleada', 'secretaria':'empleada', 'telefonista':'empleada', 'textil':'empleada', 
            'trabajadora_de_campo':'empleada', 'trabajadora_del_hogar':'empleada', 
            'trabajadora_del_sector_publico':'empleada', 'trabajadora_sexual':'trabajadora_sexual',
            'vetarinaria':'profesionista', 'voluntaria_o_trabajadora social':'empleada'}
clean_interrupcion_legal['ocupacion2'] = clean_interrupcion_legal['ocupacion'].map(dic_ocup)

# Recategorize use of contraceptive variable ('anticonceptivo')
dic_anti = {'ninguno':'ninguno', 'na':'ninguno','condon':'condon', 'condon_y_otro':'otro',
            'anillo_vaginal':'otro', 'anticoncepcion_de_emergencia':'otro', 'anticoncepcion_de_emergencia_y_otro':'otro',
            'barrera':'otro', 'calendario':'otro', 'coito_interrumpido':'otro', 'diu':'otro', 'hormonal_inyectable':'otro',
            'hormonal_oral':'otro', 'implante_subdermico':'otro', 'inyeccion':'otro', 'medicamento':'otro',
            'parche_dermico':'otro', 'pastillas_anticonceptivas':'otro', 'ritmo':'otro', 'salpingoclacia':'otro',
            'vasectomia':'otro'}
clean_interrupcion_legal['anticonceptivo2'] = clean_interrupcion_legal['anticonceptivo'].map(dic_anti)

# Update variables list by type
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(clean_interrupcion_legal)

# Data profiling for numeric variables
eda.descriptive_stats_for_numeric_vars(clean_interrupcion_legal, numeric_variables)

# Data profiling for categorical variables
eda.descriptive_stats_for_categorical_vars(clean_interrupcion_legal, string_variables)


################################################ GRAPHS ######################################################
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(20,10))
clean_interrupcion_legal.groupby(['ano_mes_ile']).count()['ano'].plot(ax=ax,color='darkblue', marker='o')
plt.title('ILE por mes \n (2016-2019)', fontsize='16')
plt.xticks(fontsize='14')
plt.yticks(fontsize='14')
plt.show()


sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(15,7))
g = clean_interrupcion_legal.groupby(['ano_mes_ile', '23_o_mayor']).count()['mes'].unstack().plot(ax=ax, marker='o', color= ['gray', 'blue'])
ax.xaxis.grid(True, which='minor')
leg = g.axes.get_legend()
leg.set_title('')
new_labels=['Menores a 23 años', '23 años o más']
for t, l in zip(leg.texts, new_labels): t.set_text(l)
plt.ylabel("")
plt.xlabel("")
plt.title("Número de ILE por mes y rango de edad\n (2016-2019)")
plt.show()


sns.set_style("darkgrid")
plt.figure(figsize = (20,10))
plot1 = sns.boxplot(y='edad', x='nile', data=clean_interrupcion_legal, palette='dark')
sns.stripplot(y='edad', x='nile', data=clean_interrupcion_legal, color="gray", jitter=0.2, size=2.5)
# Linea horizontal
plot1.axes.axhline(y = 23, ls='--', color='red')
plot1.axes.text(-0.8, 22.7, "23 años", color = 'red', fontsize=12)
plt.title('Distribución de la edad por número de ILE', fontsize='16')
plt.xlabel('Número de Interrupciones Legales (ILE)', fontsize='14')
plt.ylabel('Edad de la mujer', fontsize='14')
plt.show()



tabla_pivote = pd.pivot_table(clean_interrupcion_legal, index='edo_civil', columns='23_o_mayor', aggfunc='count')
tabla_pivote = tabla_pivote.reindex(['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado'])
tabla_pivote['edad']

ile_menor_23 = tabla_pivote['edad'][0]/sum(tabla_pivote['edad'][0])*100
ile_mayor_23 = tabla_pivote['edad'][1]/sum(tabla_pivote['edad'][1])*100

df_aux = pd.DataFrame(tabla_pivote['edad'][0]/sum(tabla_pivote['edad'][0])*100)
df_aux = pd.concat([df_aux, tabla_pivote['edad'][1]/sum(tabla_pivote['edad'][1])*100])
df_aux['23_0_mas'] = [0,0,0,0,0,0,1,1,1,1,1,1]
df_aux['edo_civil1'] = ['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado', 'casada',
                        'union_libre', 'soltera', 'divorciada', 'separada','no_especificado']
df_aux.columns = ['Porcentaje', '23_o_mayor', 'edo_civil1']
df_aux.index = df_aux.reset_index(level=0, drop=True)
df_aux.index = list(range(0,12))

g = sns.catplot(y="Porcentaje", x='edo_civil1', hue='23_o_mayor', kind="bar", data=df_aux, palette=['gray', 'blue'], height=6, aspect=1.3, legend_out=False)
g.ax.set_yticks(np.arange(0,110,10), minor=True)
leg = g.axes.flat[0].get_legend()
leg.set_title('')
new_labels=['Menores a 23 años', '23 años o más']
for t, l in zip(leg.texts, new_labels): t.set_text(l)
plt.ylabel("")
plt.xlabel("")
plt.title("Porcentaje de mujeres que recibieron una ILE por Estado Civil")
plt.show()


df_aux = clean_interrupcion_legal[clean_interrupcion_legal.edo_civil.isin(['casada', 'soltera','union_libre']), 
                                  ['edo_civil', 'nile', 'nhijos', '23_o_mayor']]
pd.pivot_table(df_aux, index=['edo_civil', 'nhijos'], columns='nile', aggfunc='count')
#sns.lmplot( y="nile", x="nhijos", data=df_aux, fit_reg=False, hue='edo_civil', legend=True)

df_aux = clean_interrupcion_legal.groupby(['entidad','23_o_mayor']).count()
df_aux = df_aux['mes'].unstack(level=1).reset_index().sort_values(by=1, ascending=False)
df_aux[0]= df_aux[0]/sum(df_aux[0])*100
df_aux[1]= df_aux[1]/sum(df_aux[1])*100
df_aux.columns = ['entidad', 'menor', 'mayor']

df_aux2 = df_aux[0:2]
df_aux3 = df_aux[2:]
df_aux2 = df_aux2.append(pd.Series(['otras_entidades',sum(df_aux3['menor']), 
                                    sum(df_aux3['mayor'])], index=df_aux2.columns), ignore_index=True)

labels = ['Ciudad \n de México', 'Estado \n de Mexico', 'Otras \n entidades']
# the label locations
x = np.arange(len(labels))  
# the width of the bars
width = 0.35  



fig = plt.figure(figsize=(20, 5))
gs = GridSpec(nrows=1, ncols=5)

ax1 = fig.add_subplot(gs[0,1])
ax1.bar(x - width/2, df_aux2['menor'], width, color='gray', label='Menores de 23 años')
ax1.bar(x + width/2, df_aux2['mayor'], width, color='blue', label='23 años o más')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_yticks(np.arange(0,110,10), minor=True)
ax1.set_title("Top de Entidad de Residencia")
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(gs[0,2:])
my_size=np.where(df_aux3['mayor']>df_aux3['menor'], 75, 30)
my_range = df_aux3['entidad']
ax2.vlines(x = my_range, ymin=df_aux3['menor'], ymax=df_aux3['mayor'], color='black', alpha=0.4)
ax2.axvline(x = 'queretaro', color='red', ls='--', linewidth=0.5)
ax2.scatter(my_range, df_aux3['menor'], color='gray', s=my_size, alpha=1, label='Menores de 23 años')
ax2.scatter(my_range, df_aux3['mayor'], color='blue', s=my_size, alpha=1 , label='23 años o más')
ax2.set_xticks(np.arange(len(my_range)))
ax2.set_xticklabels(my_range, rotation=90, horizontalalignment='right')
ax2.set_title("Desglose de Otras entidades federativas")
ax2.legend(loc='upper right')

fig.suptitle("Porcentaje de ILE segun la entidad donde residen")
plt.show()


pd.options.display.float_format = '{:,}'.format
pivot_aux = pd.pivot_table(clean_interrupcion_legal, index='escolaridad', columns='edo_civil', aggfunc='count')['mes']
pivot_aux = pivot_aux.reindex(['posgrado', 'superior', 'media_superior', 'secundaria', 'primaria','otra', 'ninguno', 'no_especificado' ])
pivot_aux = pivot_aux[['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado']]
pivot_aux
#sns.heatmap(pivot_aux)
#pd.pivot_table(clean_interrupcion_legal, index='escolaridad', columns=['23_o_mayor', 'edo_civil'], aggfunc='count')['mes']

clean_interrupcion_legal[clean_interrupcion_legal['escolaridad']=='no_especificado']

pivot_aux = pd.pivot_table(clean_interrupcion_legal, index='escolaridad', columns='ocupacion2', aggfunc='count')['mes']
pivot_aux = pivot_aux.reindex(['posgrado', 'superior', 'media_superior', 'secundaria', 'primaria',
                               'otra', 'ninguno', 'no_especificado' ])
pint(pivot_aux)


pivot_aux = pd.pivot_table(clean_interrupcion_legal, index='edo_civil', columns='ocupacion2', aggfunc='count')['mes']
pivot_aux = pivot_aux.reindex(['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado'])
print(pivot_aux)

# Graph for Graph Exploratory Data Analysis (GEDA)
sns.set_style("darkgrid")
sns.set_palette("dark")
geda.plot_by_pairs_grid(clean_interrupcion_legal, ['edad','naborto'])

geda.plot_by_pairs_grid(clean_interrupcion_legal, ['edad','nhijos'])

geda.plot_by_pairs_grid(clean_interrupcion_legal, ['edad','fsexual'])

geda.plot_bivariate_hist(clean_interrupcion_legal, 'edad', 'naborto')

geda.barplot_by_category(clean_interrupcion_legal, 'edad', 'ocupacion2')

geda.barplot_by_category(clean_interrupcion_legal, 'edad', 'edocivil_descripcion')

#######################################  SELECTION OF VARIABLES  #############################################
base_interrupcion_legal = clean_interrupcion_legal[['ano_mes_ile', '23_o_mayor', 'edo_civil', 'escolaridad', 'ocupacion2',
                                                    'menarca', 'anticonceptivo2', 'nhijos', 'naborto', 'npartos', 
                                                    'ncesarea', 'nile' ]].copy()

# Order dataset by the temporal variable 'ano_mes_ile'
base_interrupcion_legal.sort_values(by=['ano_mes_ile'], ascending=True)

##################################### SPLIT IN 'test' AND 'train' ###############################
FECHA_DE_CORTE = '2018-07-01'
mask = (base_interrupcion_legal['ano_mes_ile'] <= FECHA_DE_CORTE)
entrenamiento = base_interrupcion_legal[mask]
prueba = base_interrupcion_legal[~mask]
print('No. observaciones en entre: ', entrenamiento.shape[0], ' que representa el ', round(entrenamiento.shape[0]/base_interrupcion_legal.shape[0]*100,0),'%')
print('No. observaciones en prueb: ', prueba.shape[0], ' que representa el ', round(prueba.shape[0]/base_interrupcion_legal.shape[0]*100,0),'%')

y_entrenamiento = entrenamiento['23_o_mayor']
x_entrenamiento = pd.DataFrame(entrenamiento.drop(['23_o_mayor', 'ano_mes_ile'], axis = 1))

y_prueba = prueba['23_o_mayor']
x_prueba = pd.DataFrame(prueba.drop(['23_o_mayor', 'ano_mes_ile'], axis = 1))

################################ IMPUTATION OF VARIABLES FOR TRAINING SET ####################################

df = x_entrenamiento

var = 'edo_civil'
aux = df[[var]]
aux_imputer_edo_civil = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
aux_imputer_edo_civil.fit(aux)
aux_imputed_edo_civil = aux_imputer_edo_civil.transform(aux)
df[var] = aux_imputed_edo_civil

var = 'escolaridad'
aux = df[[var]]
aux_imputer_escolaridad = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
aux_imputer_escolaridad.fit(aux)
aux_imputed_escolaridad = aux_imputer_escolaridad.transform(aux)
df[var] = aux_imputed_escolaridad

var = 'menarca'
aux = df[[var]]
aux_imputer_menarca = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
aux_imputer_menarca.fit(aux)
aux_imputed_menarca = aux_imputer_menarca.transform(aux)
df[var] = aux_imputed_menarca

var = 'anticonceptivo2'
aux = df[[var]]
aux_imputer_anticonceptivo2 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="ninguno")
aux_imputer_anticonceptivo2.fit(aux)
aux_imputed_anticonceptivo2 = aux_imputer_anticonceptivo2.transform(aux)
df[var]=aux_imputed_anticonceptivo2

var = 'ocupacion2'
aux = df[[var]]
aux_imputer_ocupacion2 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="no_especificado")
aux_imputer_ocupacion2.fit(aux)
aux_imputed_ocupacion2 = aux_imputer_ocupacion2.transform(aux)
df[var]=aux_imputed_ocupacion2

var = 'nhijos'
aux = df[[var]]
aux_imputer_nhijos = SimpleImputer(missing_values=np.nan, strategy="median")
aux_imputer_nhijos.fit(aux)
aux_imputed_nhijos = aux_imputer_nhijos.transform(aux)
df[var]=aux_imputed_nhijos

var = 'naborto'
aux = df[[var]]
aux_imputer_naborto = SimpleImputer(missing_values=np.nan, strategy="median")
aux_imputer_naborto.fit(aux)
aux_imputed_naborto = aux_imputer_naborto.transform(aux)
df[var]=aux_imputed_naborto

var = 'npartos'
aux = df[[var]]
aux_imputer_nparto = SimpleImputer(missing_values=np.nan, strategy="median")
aux_imputer_nparto.fit(aux)
aux_imputed_nparto = aux_imputer_nparto.transform(aux)
df[var]=aux_imputed_nparto

var = 'ncesarea'
aux = df[[var]]
aux_imputer_ncesarea = SimpleImputer(missing_values=np.nan, strategy="median")
aux_imputer_ncesarea.fit(aux)
aux_imputed_ncesarea = aux_imputer_ncesarea.transform(aux)
df[var]=aux_imputed_ncesarea

var = 'nile'
aux = df[[var]]
aux_imputer_nile = SimpleImputer(missing_values=np.nan, strategy="median")
aux_imputer_nile.fit(aux)
aux_imputed_nile = aux_imputer_nile.transform(aux)
df[var]=aux_imputed_nile

# Check there is no missing values
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(x_entrenamiento)
eda.descriptive_stats_for_numeric_vars(x_entrenamiento, numeric_variables)
eda.descriptive_stats_for_categorical_vars(x_entrenamiento, string_variables)

# One-hot encoder for training set
dummies = feature_selection.transforma_cat_dummies(x_entrenamiento, string_variables, False)
x_entrenamiento = pd.concat([x_entrenamiento, dummies], axis = 1)
x_entrenamiento = x_entrenamiento.drop(string_variables, axis = 1)


############################################# FEATURE ENGINEERING ####################################################
# Use Random Forest for feature engineering
# Number of trees in the forest
N_ARBOLES = 10000  
# Seed used by the random number generator
SEMILLA = 104 

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators = N_ARBOLES, random_state = SEMILLA, n_jobs=-1)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_entrenamiento, y_entrenamiento)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

feature_imp = pd.Series(clf.feature_importances_, index=x_entrenamiento.columns).sort_values(ascending=False)


sns.set_style("darkgrid")
pp = sns.cubehelix_palette(30, start=.1, rot=-.75, reverse=True)
fig = plt.figure(figsize=(8, 12))
#Visualizing Important Features
g = sns.barplot(x=feature_imp, y=feature_imp.index, palette=pp)
# Add labels to your graph
g.axes.axvline(x = 0.04, ls='--', color='red')
plt.xlabel('Score')
plt.ylabel('Variables')
plt.title("Vizualizacion del score de importancia de las variables")
plt.show()

################################ SELECTION OF VARIABLES FOR MODEL #############################################
x_entrenamiento = x_entrenamiento[['menarca', 'nhijos', 'ocupacion2_estudiante', 'npartos', 'escolaridad_superior',
                                   'ocupacion2_empleada', 'ncesarea']]

########################################## TEMPORAL CROSS VALIDATION #################################################
tscv = TimeSeriesSplit(n_splits=10)
print(tscv)

X = x_entrenamiento
y = y_entrenamiento

cv = [(train_index, test_index)
      for train_index, test_index in tscv.split(X)]

############################################### MODEL 1: DECISION TREE #################################################
classifier = tree.DecisionTreeClassifier()
# Hyperparameters
hyper_param_grid1 = {'max_depth': [1,5,10,20,50,100,500,1000], 
                     'min_samples_split': [2,5,10,20,50],
                     'criterion':['gini', 'entropy'],
                     'splitter':['random', 'best'],
                     'presort':[False, True]
                    }
# Grid search!
# verbose for debugging, '3' allows for messages
grid_search = GridSearchCV(classifier, 
                           hyper_param_grid1, 
                           scoring = 'f1',
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3
                          )
grid_search.fit(x_entrenamiento, y_entrenamiento)
# Best params
grid_search.best_params_
# Best score, average of scores in cv sets
print('Valor de la mejor métrica para el modelo de Decision Tree:', grid_search.best_score_ * 100)
# Save results
cv_results_arbol = pd.DataFrame(grid_search.cv_results_)
cv_results_arbol.to_csv("Output/DecisionTree.csv", index=False)

############################################### MODEL 2: RANDOM FOREST ###############################################
classifier = RandomForestClassifier()
# Hyperparameters
hyper_param_grid2 = {'n_estimators': [100, 500, 1000, 5000], 
                    'max_depth': [5,10,20,50,100], 
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10,20,50]
                    }
# Gid search
grid_search = GridSearchCV(classifier, 
                           hyper_param_grid2, 
                           scoring = 'f1',
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3
                          )
grid_search.fit(x_entrenamiento, y_entrenamiento)
# Best params
grid_search.best_params_
# Best score
print('Valor de la mejor métrica para el modelo de Random Forest:', grid_search.best_score_ * 100)
#Save to file
cv_results_forest = pd.DataFrame(grid_search.cv_results_)
cv_results_forest.to_csv("Output/RandomForest.csv", index=False)

############################################### MODEL 3: XGBOOST ########################################################
classifier = xgb.XGBClassifier()
# Hyperparameters
hyper_param_grid = {'max_depth': [5,10,20,50,100],
                    'min_child_weight': [2,5,10,20,50],
                    'learning_rate':[0.01,0.05,0.1],
                    'objective':['binary:logistic', 'binary:logitraw','binary:hinge']
                    }
# Grid search!
grid_search = GridSearchCV(classifier, 
                           hyper_param_grid, 
                           scoring = 'f1',
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3
                          )
grid_search.fit(x_entrenamiento, y_entrenamiento)
# Best params
grid_search.best_params_
# Best score
print('Valor de la mejor métrica para el modelo de XGBoost:', grid_search.best_score_ * 100)
# Save to file
cv_results_XGBoost = pd.DataFrame(grid_search.cv_results_)
cv_results_XGBoost.to_csv("Output/XGBoost.csv", index=False)

####################################### SELECTION OF MODEL AND HYPERPARAMETERS #######################################
results_all = pd.concat([cv_results_forest, cv_results_arbol, cv_results_XGBoost],
                        axis=0, sort='False').sort_values(by='rank_test_score', ascending=True).head()
results_all.to_csv("Output/all_models.csv", index=False)
# Best parameters according to specifided metric
results_all.iloc[0,:].params

################################## PREDICTION FOR 'test' SET ################################################
# Imputation for test set
df = x_prueba

var = 'edo_civil'
aux = df[[var]]
aux_imputed_edo_civil = aux_imputer_edo_civil.transform(aux)
df[var] = aux_imputed_edo_civil

var = 'escolaridad'
aux = df[[var]]
aux_imputed_escolaridad = aux_imputer_escolaridad.transform(aux)
df[var] = aux_imputed_escolaridad

var = 'menarca'
aux = df[[var]]
aux_imputed_menarca = aux_imputer_menarca.transform(aux)
df[var] = aux_imputed_menarca

var = 'anticonceptivo2'
aux = df[[var]]
aux_imputed_anticonceptivo2 = aux_imputer_anticonceptivo2.transform(aux)
df[var]=aux_imputed_anticonceptivo2

var = 'ocupacion2'
aux = df[[var]]
aux_imputed_ocupacion2 = aux_imputer_ocupacion2.transform(aux)
df[var]=aux_imputed_ocupacion2

var = 'nhijos'
aux = df[[var]]
aux_imputed_nhijos = aux_imputer_nhijos.transform(aux)
df[var]=aux_imputed_nhijos

var = 'naborto'
aux = df[[var]]
aux_imputed_naborto = aux_imputer_naborto.transform(aux)
df[var]=aux_imputed_naborto

var = 'npartos'
aux = df[[var]]
aux_imputed_nparto = aux_imputer_nparto.transform(aux)
df[var]=aux_imputed_nparto

var = 'ncesarea'
aux = df[[var]]
aux_imputed_ncesarea = aux_imputer_ncesarea.transform(aux)
df[var]=aux_imputed_ncesarea

var = 'nile'
aux = df[[var]]
aux_imputed_nile = aux_imputer_nile.transform(aux)
df[var]=aux_imputed_nile

# One-hot encoder for test set
dummies = feature_selection.transforma_cat_dummies(x_prueba, string_variables, False)
x_prueba = pd.concat([x_prueba, dummies], axis = 1)
x_prueba = x_prueba.drop(string_variables, axis = 1)

# Variables for test set
x_prueba = x_prueba[['menarca', 'nhijos', 'ocupacion2_estudiante', 'npartos',
                     'escolaridad_superior', 'ocupacion2_empleada', 'ncesarea']]

################################################# SELECTED MODEL ##############################################

# Specify parameters via map
param_finales = {'learning_rate': 0.05,
                 'max_depth': 10,
                 'min_child_weight': 10,
                 'objective': 'binary:hinge'}
NUM_ROUNDS = 5000
xgdmat = xgb.DMatrix(x_entrenamiento, y_entrenamiento)
best_model = xgb.train(param_finales, xgdmat, NUM_ROUNDS)


######################################## PREDICTION FOR "test" SET #####################################
testdmat = xgb.DMatrix(x_prueba)
y_predict = best_model.predict(testdmat)

###################################### SELECTED MODEL METRICS ################################################
print('Accuracy:', accuracy_score(y_predict, y_prueba))
print('Recall:', recall_score(y_prueba, y_predict, average='weighted'))
print('F1 score:', f1_score(y_prueba, y_predict, average='weighted'))

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_prueba, y_predict)
roc_auc = auc(fpr, tpr)

sns.set_style("darkgrid")
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix
conf_mat = pd.DataFrame(confusion_matrix(y_prueba, y_predict))
conf_mat = conf_mat[conf_mat.columns[::-1]].iloc[::-1]
print(conf_mat)
conf_mat.div(conf_mat.sum(0), axis=1).round(2)*100