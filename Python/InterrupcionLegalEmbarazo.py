#!/usr/bin/env python
# coding: utf-8

# # Interrupcion legal del embarazo CDMX
# 

#import python packages
import pandas as pd
import numpy as np
import pandas_profiling
from datetime import datetime, date, time, timedelta
# import calendar
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as dates
from matplotlib.gridspec import GridSpec


#import functions created in .py
import load_data 
import clean_data
import transform_data
import eda
import geda
import feature_selection



##################################################  CARGAR BASE  #############################################################
#Update data file name
datafile_name = 'interrupcion-legal-del-embarazo.csv'
#Update path to data file

path_to_datafile = '/home/bj/Documents/IntroDataScience/Proyecto/Data'
#path_to_datafile ='C:/Users/GZARAZUA/Desktop/MCD/MINERIA DE DATOS/Reporte aborto'

#we load the database
interrupcion_legal = load_data.load_df(datafile_name, path_to_datafile, ';')



#We define a maximum number of columns to display
pd.options.display.max_columns = 80
#We take a first look at the database
interrupcion_legal.head()

########################################  LIMPIAR NOMBRE VARIABLES  ######################################################
#Standardization of variable names
clean_interrupcion_legal=clean_data.clean_variables_names(interrupcion_legal)
#Wlet's look at the results
clean_interrupcion_legal.head()


#general data frame information
eda.df_variables_info(clean_interrupcion_legal)
#variables that will contain the type of existing variables
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(clean_interrupcion_legal)
#the classification of the variables is displayed
eda.print_info_type_of_vars(numeric_variables, categorical_variables, dates_variables, string_variables)


#######################################  LIMPIAMOS VALORES DE VARIABLES  ###############################################
#Se observa que de las variables de tipo "string", 3 corresponden a variables tipo "date" que
#no fueron reconocidas, a saber: fingreso, fmenstrua, h_fingreso
#Primeramente estandarizaremos los valores del resto de las variables string
clean_interrupcion_legal=transform_data.clean_variables_values(clean_interrupcion_legal, ['mes','autoref','edocivil_descripcion','desc_derechohab','nivel_edu','ocupacion','religion','parentesco','entidad','alc_o_municipio','consejeria','anticonceptivo','motiles','desc_servicio','p_consent','procile','s_complica','panticoncep'
])
clean_interrupcion_legal.head(30)


#######################################  HACEMOS RECONOCER VARIABLES DE FECHA  ###########################################
#Corrección fmenstrua...

#now we proceed to recognize the date type variable that were not initially recognized
#starting with 'fmenstrua'
clean_interrupcion_legal['fmenstrua'] = pd.to_datetime(clean_interrupcion_legal['fmenstrua'])


#Corrección fingreso...

#We change all the np.nan of fingreso that pd.read_csv detected
clean_interrupcion_legal['fingreso']=clean_interrupcion_legal['fingreso'].replace(np.nan,"NA")
#We implemented a provisional function to divide the data by '/' 
#so that we have a column with day, another with month and another with year.
#The intention is to verify that it has no errors
dividir=lambda x: pd.Series([i for i in x.split('/')])
#we create a provisional basis for the day, month and year data
aux=clean_interrupcion_legal['fingreso'].apply(dividir)
#we return the variable of 'pretend' with the original values (with the np.nan)
clean_interrupcion_legal['fingreso']=clean_interrupcion_legal['fingreso'].replace("NA",np.nan)


aux=pd.DataFrame(aux)
aux.columns=['dia','mes','ano']
#We look for months that are longer than 12 months (which is a mistake)
mes_mayor_12=aux[pd.to_numeric(aux['mes'])>12]
print('Observaciones con mes mayor a 12: \n',mes_mayor_12,'\n')
#We look for data whose year is less than 2016, as it could be an error
ano_menor_16_mayor_19=aux[(pd.to_numeric(aux['ano'])<16) | (pd.to_numeric(aux['ano'])>19)]
print('Observaciones con año menor a 2016 o mayor a 2019: \n', ano_menor_16_mayor_19)



#let's get the no. of observation to correct it
no_observ=mes_mayor_12.loc[mes_mayor_12['mes']=='0719'].index.values[0]
clean_interrupcion_legal['fingreso'][no_observ]
#correct data
clean_interrupcion_legal['fingreso'][no_observ]='17/07/19'
#check the correction
#clean_interrupcion_legal['fingreso'][no_observ]


#Se asjuta el año de la fecha 'fingreso' acorde a la variable 'ano' (se asume que el año difiere por ser un error)
temp=ano_menor_16_mayor_19.index.values
for i in range(0,len(temp)):
    #correct data
    clean_interrupcion_legal['fingreso'][temp[i]]=ano_menor_16_mayor_19.iloc[i,0]+'/'+ano_menor_16_mayor_19.iloc[i,1]+'/'+str(clean_interrupcion_legal['ano'][temp[i]])[-2:]
    #check the correction
    #print(clean_interrupcion_legal['fingreso'][temp[i]])


#now we proceed to recognize the date type variable
clean_interrupcion_legal['fingreso'] = pd.to_datetime(clean_interrupcion_legal['fingreso'])


#variables that will contain the type of existing variables
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(clean_interrupcion_legal)
#the classification of the variables is displayed
eda.print_info_type_of_vars(numeric_variables, categorical_variables, dates_variables, string_variables)


#######################################  CORRECCIÓN DE ALGUNOS ERRORES  ###########################################
#Corrección/observación autoref...

#most of the categories chosen are not representative; It is noted that the category
#autorreferida y autoreferida are significant and assumed to be equal, so
#will be defined with the same name
clean_interrupcion_legal.loc[clean_interrupcion_legal['autoref']=='autoreferida', 'autoref']='autorreferida'


#Corrección/observación ocupacion...

clean_interrupcion_legal.loc[clean_interrupcion_legal['ocupacion']=='constructura_o_elecetricista', 'ocupacion']='constructora_o_elecetricista'



#Se observa una fecha fmenstrua mayor a 2019, lo cual es un error; se cambia por NaT
clean_interrupcion_legal.loc[clean_interrupcion_legal['fmenstrua'] > pd.Timestamp(date(2019,12,31)), 'fmenstrua']=pd.to_datetime(np.nan)



#Creamos la variable objetivo
clean_interrupcion_legal['23_o_mayor'] = np.where(clean_interrupcion_legal['edad'] >= 23, 1, 0)
clean_interrupcion_legal.head()

##Concatenamos las variables ano y mes (convirtiendo mes a numero)
##transform_data.month_string_to_number('mar')
mes_num = clean_interrupcion_legal['mes'].map(transform_data.month_string_to_number_string)


clean_interrupcion_legal['ano_mes_ile'] = clean_interrupcion_legal['ano'].astype(str) + "/" + mes_num.astype(str)

clean_interrupcion_legal['ano_mes_ile'] = pd.to_datetime(clean_interrupcion_legal['ano_mes_ile'])


#Edad por quintiles
clean_interrupcion_legal['edad_quintiles'] = pd.cut(clean_interrupcion_legal['edad'], bins=[0,4,9,14,19,24,29,34,39,44,49,54,59,100], labels=["0 a 4", "5 a 9", "10 a 14", "15 a 19", "20 a 24", "25 a 29", "30 a 34", "35 a 39", "40 a 44", "45 a 49", "50 a 54", "55 a 59", "60 y mas"], right=True)


#Recategorizacion de *estado_civil*
dic_edo = {'soltera':'soltera', 'divorciada':'divorciada', 'union_libre':'union_libre', 'casada':'casada', 'separada':'separada', 'n/e': 'no_especificado', 'na':'no_especificado'}
clean_interrupcion_legal['edo_civil'] = clean_interrupcion_legal['edocivil_descripcion'].map(dic_edo)


#Recategorizacion de *nivel_educativo*
dic_escol = { 'ninguno':'ninguno', 'preescolar':'ninguno', 'primaria_completa':'primaria', 'primaria_incompleta':'primaria',
             'secundaria_incompleta':'secundaria', 'secundaria_completa':'secundaria', 'preparatoria_incompleta':'media_superior',
             'preparatoria_completa':'media_superior', 'carrera_tecnica':'superior', 'licenciatura_incompleta':'superior','licenciatura_completa':'superior',
             'maestria':'posgrado', 'doctorado':'posgrado', 'posgrado':'posgrado', 'posgrado_incompleto':'posgrado',
             'no_especifica':'no_especificado', 'NA':'no_especificado', 'otra':'otra'}

clean_interrupcion_legal['escolaridad'] = clean_interrupcion_legal['nivel_edu'].map(dic_escol)


#Recategorizacion de *ocupacion*
dic_ocup = {'abogada':'profesionista', 'administradora':'profesionista', 'ama_de_casa':'ama_de_casa', 'arquitecta':'profesionista',
            'artista':'profesionista', 'asesora_financiera':'profesionista', 'asesora_juridica':'profesionista', 'auditora':'profesionista',
            'bibliotecaria':'profesionista', 'cajera':'empleada', 'capturista':'empleada', 'chofer':'empleada', 'cientifica':'profesionista',
            'cocinera':'empleada', 'comerciante':'empleada', 'constructora_o_elecetricista':'empleada', 'constructura_o_elecetricista':'empleada',
            'contadora':'empleada', 'dentista':'profesionista', 'desempleada':'desempleada', 'diseñadora':'profesionista',
            'ejecutiva':'profesionista', 'empleada':'empleada', 'enfermera':'profesionista', 'estilista':'empleada',
            'estudiante':'estudiante', 'fisioterapeuta':'empleada', 'fotografa':'empleada', 'informatica_o_tecnologia':'profesionista',
            'ingeniera':'profesionista', 'logistica_o_eventos':'empleada', 'mecanica':'profesionista', 'medico':'profesionista',
            'mesera':'empleada', 'modelo':'empleada', 'na':'no_especificado', 'nutiologa':'profesionista', 'obrera':'empleada',
            'paramedico':'empleada', 'periodista_o_publicista':'profesionista', 'policia_o_seguridad':'empleada',
            'profesora_o_educadora':'profesionista', 'psicologa':'profesionista', 'recepcionista':'empleada', 'recursos_humanos':'empleada',
            'secretaria':'empleada', 'telefonista':'empleada', 'textil':'empleada', 'trabajadora_de_campo':'empleada',
            'trabajadora_del_hogar':'empleada', 'trabajadora_del_sector_publico':'empleada', 'trabajadora_sexual':'trabajadora_sexual',
            'vetarinaria':'profesionista', 'voluntaria_o_trabajadora social':'empleada'}
clean_interrupcion_legal['ocupacion2'] = clean_interrupcion_legal['ocupacion'].map(dic_ocup)


#Recategorizacion de *anticonceptivo*
dic_anti = { 'ninguno':'ninguno', 'na':'ninguno','condon':'condon', 'condon_y_otro':'otro',
            'anillo_vaginal':'otro', 'anticoncepcion_de_emergencia':'otro', 'anticoncepcion_de_emergencia_y_otro':'otro',
            'barrera':'otro', 'calendario':'otro', 'coito_interrumpido':'otro', 'diu':'otro', 'hormonal_inyectable':'otro',
            'hormonal_oral':'otro', 'implante_subdermico':'otro', 'inyeccion':'otro', 'medicamento':'otro',
            'parche_dermico':'otro', 'pastillas_anticonceptivas':'otro', 'ritmo':'otro', 'salpingoclacia':'otro',
            'vasectomia':'otro'}
clean_interrupcion_legal['anticonceptivo2'] = clean_interrupcion_legal['anticonceptivo'].map(dic_anti)


#variables that will contain the type of existing variables
[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(clean_interrupcion_legal)
#the classification of the variables is displayed
eda.print_info_type_of_vars(numeric_variables, categorical_variables, dates_variables, string_variables)

eda.descriptive_stats_for_numeric_vars(clean_interrupcion_legal, numeric_variables)

eda.descriptive_stats_for_categorical_vars(clean_interrupcion_legal, categorical_variables)

clean_interrupcion_legal.head()


################################################# GEDA ########################################################

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(20,10))
clean_interrupcion_legal.groupby(['ano_mes_ile']).count()['ano'].plot(ax=ax,color='darkblue', marker='o')

plt.title('ILE por mes \n (2016-2019)', fontsize='16')
plt.xticks(fontsize='14')
plt.yticks(fontsize='14')
plt.show()


# In[120]:


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


# In[121]:


sns.set_style("darkgrid")
plt.figure(figsize = (20,10))
#plt.style.use('ggplot')
plot1 = sns.boxplot(y='edad', x='nile', data=clean_interrupcion_legal, palette='dark')
sns.stripplot(y='edad', x='nile', data=clean_interrupcion_legal, color="gray", jitter=0.2, size=2.5)
# Linea horizontal
plot1.axes.axhline(y = 23, ls='--', color='red')
plot1.axes.text(-0.8, 22.7, "23 años", color = 'red', fontsize=12)

plt.title('Distribución de la edad por número de ILE', fontsize='16')
plt.xlabel('Número de Interrupciones Legales (ILE)', fontsize='14')
plt.ylabel('Edad de la mujer', fontsize='14')

plt.show()


# In[122]:


tabla_pivote = pd.pivot_table(clean_interrupcion_legal, index='edo_civil', columns='23_o_mayor', aggfunc='count')
tabla_pivote = tabla_pivote.reindex(['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado'])
tabla_pivote['edad']


# In[123]:


ile_menor_23 = tabla_pivote['edad'][0]/sum(tabla_pivote['edad'][0])*100
ile_mayor_23 = tabla_pivote['edad'][1]/sum(tabla_pivote['edad'][1])*100


# In[124]:


df_aux = pd.DataFrame(tabla_pivote['edad'][0]/sum(tabla_pivote['edad'][0])*100)
df_aux = pd.concat([df_aux, tabla_pivote['edad'][1]/sum(tabla_pivote['edad'][1])*100])
df_aux['23_0_mas'] = [0,0,0,0,0,0,1,1,1,1,1,1]
df_aux['edo_civil1'] = ['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado', 'casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado']
df_aux.columns = ['Porcentaje', '23_o_mayor', 'edo_civil1']
df_aux.index = df_aux.reset_index(level=0, drop=True)
df_aux.index = [0,1,2,3,4,5,6,7,8,9,10,11]
#
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


# En la presente gráfica se observan diferencias entre las mujeres de 23 años o más y las de menos, acorde a estado civil, lo cual sugiere que el edo civil tiene 

# In[125]:


df_aux = clean_interrupcion_legal[clean_interrupcion_legal.edo_civil.isin(['casada', 'soltera','union_libre'])][['edo_civil', 'nile', 'nhijos', '23_o_mayor']]
pd.pivot_table(df_aux, index=['edo_civil', 'nhijos'], columns='nile', aggfunc='count')
#sns.lmplot( y="nile", x="nhijos", data=df_aux, fit_reg=False, hue='edo_civil', legend=True)


# In[126]:


df_aux = clean_interrupcion_legal.groupby(['entidad', '23_o_mayor']).count()['mes'].unstack(level=1).reset_index().sort_values(by=1, ascending=False)
df_aux[0]= df_aux[0]/sum(df_aux[0])*100
df_aux[1]= df_aux[1]/sum(df_aux[1])*100
df_aux.columns = ['entidad', 'menor', 'mayor']

df_aux2 = df_aux[0:2]
df_aux3 = df_aux[2:]
df_aux2 = df_aux2.append(pd.Series(['otras_entidades',sum(df_aux3['menor']), sum(df_aux3['mayor'])], index=df_aux2.columns), ignore_index=True)

#labels = df_aux2['entidad']
labels = ['Ciudad \n de México', 'Estado \n de Mexico', 'Otras \n entidades']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars



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
#format_axes(fig)

plt.show()


# In[127]:


pd.options.display.float_format = '{:,}'.format
pivot_aux = pd.pivot_table(clean_interrupcion_legal, index='escolaridad', columns='edo_civil', aggfunc='count')['mes']
pivot_aux = pivot_aux.reindex(['posgrado', 'superior', 'media_superior', 'secundaria', 'primaria','otra', 'ninguno', 'no_especificado' ])
pivot_aux = pivot_aux[['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado']]
pivot_aux
#sns.heatmap(pivot_aux)
#pd.pivot_table(clean_interrupcion_legal, index='escolaridad', columns=['23_o_mayor', 'edo_civil'], aggfunc='count')['mes']


# In[128]:


clean_interrupcion_legal[clean_interrupcion_legal['escolaridad']=='no_especificado']


# In[129]:


pivot_aux = pd.pivot_table(clean_interrupcion_legal, index='escolaridad', columns='ocupacion2', aggfunc='count')['mes']
pivot_aux = pivot_aux.reindex(['posgrado', 'superior', 'media_superior', 'secundaria', 'primaria','otra', 'ninguno', 'no_especificado' ])
pivot_aux 


# In[130]:


pivot_aux = pd.pivot_table(clean_interrupcion_legal, index='edo_civil', columns='ocupacion2', aggfunc='count')['mes']
pivot_aux = pivot_aux.reindex(['casada', 'union_libre', 'soltera', 'divorciada', 'separada','no_especificado'])
pivot_aux 


geda.plot_by_pairs_grid(clean_interrupcion_legal, ['edad','naborto'])


# In[135]:


geda.plot_by_pairs_grid(clean_interrupcion_legal, ['edad','nhijos'])


# In[136]:


geda.plot_by_pairs_grid(clean_interrupcion_legal, ['edad','fsexual'])


# In[137]:


geda.plot_bivariate_hist(clean_interrupcion_legal, 'edad', 'naborto')


# In[138]:


geda.barplot_by_category(clean_interrupcion_legal, 'edad', 'ocupacion2')


# In[139]:


geda.barplot_by_category(clean_interrupcion_legal, 'edad', 'ocupacion2')


# In[140]:


geda.barplot_by_category(clean_interrupcion_legal, 'edad', 'edocivil_descripcion')


############################################# Separacion de datos de *entrenamiento* y *prueba* ####################################

base_interrupcion_legal = clean_interrupcion_legal[['ano_mes_ile', '23_o_mayor', 'edo_civil', 'escolaridad', 'ocupacion2', 'menarca', 'anticonceptivo2', 'nhijos', 'naborto', 'npartos', 'ncesarea', 'nile' ]].copy()

#Ordenamos los datos conforme a la variable temporal
base_interrupcion_legal.sort_values(by=['ano_mes_ile'], ascending=True)


#Lo deje hasta julio de 2018 poara tener un anio de prueba
mask = (base_interrupcion_legal['ano_mes_ile']<='2018-07-01')
entrenamiento = base_interrupcion_legal[mask]
prueba = base_interrupcion_legal[~mask]
print('No. observaciones en entre: ', entrenamiento.shape[0], ' que representa el ', round(entrenamiento.shape[0]/base_interrupcion_legal.shape[0]*100,0),'%')
print('No. observaciones en prueb: ', prueba.shape[0], ' que representa el ', round(prueba.shape[0]/base_interrupcion_legal.shape[0]*100,0),'%')


# In[242]:


y_entrenamiento = entrenamiento['23_o_mayor']
x_entrenamiento = pd.DataFrame(entrenamiento.drop(['23_o_mayor', 'ano_mes_ile'], axis = 1))

y_prueba = prueba['23_o_mayor']
x_prueba = pd.DataFrame(prueba.drop(['23_o_mayor', 'ano_mes_ile'], axis = 1))


##################################################### Imputacion de variables ########################################

# In[144]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer


# In[145]:


#Imputacion para base de entrenamiento

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


# Checamos que, en efecto no haya missings

# In[146]:


eda.df_variables_info(x_entrenamiento)

[numeric_variables, categorical_variables, dates_variables, string_variables] = eda.info_type_of_vars(x_entrenamiento)

print(numeric_variables)
print(categorical_variables)
print(dates_variables)
print(string_variables)


# In[147]:


eda.descriptive_stats_for_numeric_vars(x_entrenamiento, numeric_variables)
eda.descriptive_stats_for_categorical_vars(x_entrenamiento, string_variables)


##################################### One-hot encoder para vars categoricas ####################################################
### One-hot encoder Para Entrenamiento
dummies = feature_selection.transforma_cat_dummies(x_entrenamiento, string_variables, False)
x_entrenamiento = pd.concat([x_entrenamiento, dummies], axis = 1)
x_entrenamiento = x_entrenamiento.drop(string_variables, axis = 1)


# In[149]:


x_entrenamiento.head()

##################################################### Feature Engineering  ################################################
# ## Random forest para parametros

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[151]:


N_ARBOLES = 1000  #The number of trees in the forest
SEMILLA = 104 # the seed used by the random number generator

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
feature_imp


# In[152]:


sns.set_style("darkgrid")
pp = sns.cubehelix_palette(30, start=.1, rot=-.75, reverse=True)
fig = plt.figure(figsize=(8, 12))
#Visualizing Important Features
g = sns.barplot(x=feature_imp, y=feature_imp.index, palette=pp)
# Add labels to your graph
#g.axes.axhline(y = 'naborto', ls='--', color='red')
plt.xlabel('Score')
plt.ylabel('Variables')
plt.title("Vizualizacion del score de importancia de las variables")
plt.show()


# Nos quedamos con las siguientes variables:
# - menarca
# - nhijos
# - ocupacion2_estudiante
# - npartos
# - escolaridad_superior
# - ocupacion2_empleada
# - ncesarea
# - naborto

#  Para entrenamiento
x_entrenamiento = x_entrenamiento[['menarca', 'nhijos', 'ocupacion2_estudiante', 'npartos', 'escolaridad_superior', 'ocupacion2_empleada', 'ncesarea', 'naborto']]


############################################################# Hiperparametros (Magic Loop)  ###############################


from sklearn.model_selection import TimeSeriesSplit #Para split temporal
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, auc, roc_curve
from sklearn import tree
import xgboost as xgb


# In[160]:


tscv = TimeSeriesSplit(n_splits=10)

print(tscv)

X = x_entrenamiento
y = y_entrenamiento

cv = [(train_index, test_index)
      for train_index, test_index in tscv.split(X)]
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
#    y_train, y_test = y.iloc[train_index], y.iloc[test_index]   
cv


####################################################### 1. Random Forest

# In[162]:


#ocuparemos un RF
classifier = RandomForestClassifier()

##separando en train, test
#X_train, X_test, y_train, y_test = train_test_split(X, y)

#definicion de los hiperparametros que queremos probar
hyper_param_grid1 = {'n_estimators': [1,10,100,1000], 
                    'max_depth': [1,5,10,20,50,100], 
                    'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10]}

#ocupemos grid search!, el verbose permite hacer debugging ... el 3 nos permitirá ver los mensajes de 3 trials de los 10 
grid_search = GridSearchCV(classifier, 
                           hyper_param_grid1, 
                           scoring = 'f1',
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3)
grid_search.fit(x_entrenamiento, y_entrenamiento)

#de los valores posibles que pusimos en el grid, cuáles fueron los mejores
grid_search.best_params_

#mejor score asociado a los modelos generados con los diferentes hiperparametros
#corresponde al promedio de los scores generados con los cv
grid_search.best_score_


# In[163]:


cv_results_forest = pd.DataFrame(grid_search.cv_results_)
cv_results_forest.head()
#Save to file
cv_results_forest.to_csv(index=False)


#################################################### 2. Arbol

# In[172]:


#ocuparemos un RF
classifier = tree.DecisionTreeClassifier()

##separando en train, test
#X_train, X_test, y_train, y_test = train_test_split(X, y)

#definicion de los hiperparametros que queremos probar
hyper_param_grid2 = {'max_depth': [1,5,10,20,50,100], 
                     'min_samples_split': [2,5,10],
                    'criterion':['gini', 'entropy'],
                    'splitter':['random', 'best'],
                    'presort':[False, True]}

#ocupemos grid search!, el verbose permite hacer debugging ... el 3 nos permitirá ver los mensajes de 3 trials de los 10 
grid_search = GridSearchCV(classifier, 
                           hyper_param_grid2, 
                           scoring = 'f1',
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3)
grid_search.fit(x_entrenamiento, y_entrenamiento)

#de los valores posibles que pusimos en el grid, cuáles fueron los mejores
grid_search.best_params_

#mejor score asociado a los modelos generados con los diferentes hiperparametros
#corresponde al promedio de los scores generados con los cv
grid_search.best_score_


# In[173]:


cv_results_arbol = pd.DataFrame(grid_search.cv_results_)
cv_results_arbol.head()
#
cv_results_arbol.to_csv(index=False)


####################################################### 3. XGBoost

# In[177]:


#ocuparemos un RF
classifier = xgb.XGBClassifier()

##separando en train, test
#X_train, X_test, y_train, y_test = train_test_split(X, y)

#definicion de los hiperparametros que queremos probar
hyper_param_grid = {'max_depth': [1,5,10,20,50,100],
                    'min_samples_split': [2,5,10],
                    'booster':['gbtree', 'gblinear', 'dart']
                    }

#ocupemos grid search!, el verbose permite hacer debugging ... el 3 nos permitirá ver los mensajes de 3 trials de los 10 
grid_search = GridSearchCV(classifier, 
                           hyper_param_grid, 
                           scoring = 'f1',
                           cv = cv, 
                           n_jobs = -1,
                           verbose = 3)
grid_search.fit(x_entrenamiento, y_entrenamiento)

#de los valores posibles que pusimos en el grid, cuáles fueron los mejores
grid_search.best_params_

#mejor score asociado a los modelos generados con los diferentes hiperparametros
#corresponde al promedio de los scores generados con los cv
grid_search.best_score_


# In[189]:


cv_results_XGBoost = pd.DataFrame(grid_search.cv_results_)
cv_results_XGBoost.head()
#
cv_results_XGBoost.to_csv(index=False)


# Juntamos los resultados de todos los modelos

# In[183]:


results_all = pd.concat([cv_results_forest, cv_results_arbol, cv_results_XGBoost], axis=0, sort='False').sort_values(by='rank_test_score', ascending=True).head()

## mejor configuración de hiperparámetros de acuerdo a nuestra métrica
results_all.iloc[0,:].params


# In[191]:


#cv_results_forest
#cv_results_arbol
cv_results_XGBoost


############################################ Seleccion de modelo  y aplicacion en muestra de prueba #################################
#######  Imputacion para base de prueba

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


########## One -hot encoder Para Prueba
dummies = feature_selection.transforma_cat_dummies(x_prueba, string_variables, False)
x_prueba = pd.concat([x_prueba, dummies], axis = 1)
x_prueba = x_prueba.drop(string_variables, axis = 1)


######### Seleccion de variables Para entrenamiento
x_prueba = x_prueba[['menarca', 'nhijos', 'ocupacion2_estudiante', 'npartos', 'escolaridad_superior', 'ocupacion2_empleada', 'ncesarea', 'naborto']]


########## Definicion de los parametros del modelo
# specify parameters via map
param_finales = {'max_depth':1, 'min_samples_split': 10, 'booster':'gbtree'}

num_round = 500

xgdmat = xgb.DMatrix(x_entrenamiento, y_entrenamiento)

best_model = xgb.train(param_finales, xgdmat, num_round)


#we can then plot our feature importances using a built-in method. This is similar to the feature importances found in sklearn.
xgb.plot_importance(best_model)


####################################### Prediccion para la muestra de prueba ###############################################

# In[232]:


testdmat = xgb.DMatrix(x_prueba)
y_predict = best_model.predict(testdmat)
y_predict


# In[233]:


# Convertimos a etiquetas 0/1
THRESHOLD = 0.5

y_predict[y_predict > THRESHOLD] = 1
y_predict[y_predict <= THRESHOLD] = 0
y_predict

##################################################### Metricas del modelo  ####################################################

# Acuracy en la muestra de prueba

# In[234]:


accuracy_score(y_predict, y_prueba)


# In[235]:


recall_score(y_prueba, y_predict, average='weighted')


# In[236]:


f1_score(y_prueba, y_predict, average='weighted')


# In[237]:


##################################################### ROC curve ############################################################
fpr, tpr, thresholds = metrics.roc_curve(y_prueba, y_predict)
roc_auc = auc(fpr, tpr)


# In[238]:


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


################################################  Confusion matrix ########################################################

print(confusion_matrix(y_prueba, y_predict))
