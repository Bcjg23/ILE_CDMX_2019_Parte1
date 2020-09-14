
# Interrupcion Legal del Embarazo

El siguiente proyecto se refiere al análisis de datos de la base de interrupcion legal del embarazo con datos de 2016-2019, obtenida de la [página oficial](https://datos.cdmx.gob.mx/explore/dataset/interrupcion-legal-del-embarazo).

## Prerequisitos

Se tiene que tener instalado Python 3. Las versiones específicas de paquetes se encuentran recopiladas en el archivo **Requirements.txt**. Estos requerimientos tienen que ser instalados para poder correr el análisis.

## Instalación

El paquete de archivos desarrollados son los siguientes:
- load_data 
- clean_data
- transform_data
- eda
- geda
- feature_selection

El archivo principal para correr el análisis de datos es el archivo **main_file.py**. A este archivo se le tienen que hacer los cambios para especificar la ruta del archivo que contiene los datos y su ruta. 

Se deben seguir los siguientes pasos:

 1. Descargar la base de datos desde [página oficial] con un nombre *my_data.csv* 
 2. Todos los archivos del paquete deben ser descargados, no necesariamente al mismo folder
 3. Abrir el archivo *InterrupcionLegalEmbarazo.py* y realizar las siguientes acualizaciones:
      3.1 Actualizar el nombre del archivo en la linea indicada por **# Update data file name**. Nota que el nombre se debe encontrar entre comillas, por ejemplo *"interrupcion-legal-del-embarazo.csv"*.
      3.2. Actualizar la ruta donde se encuentra el archivo *interrupcion-legal-del-embarazo.csv*, indicada en la linea **# Update path to data file**. Por ejemplo, puedes actualizar la ruta como *"/home/bj/Documents/IntroDataScience/eda_hw/Data"*. Observe que la ruta no contiene el **/** final.
      

## Reporte

https://bcjg23.github.io/ILE_CDMX_2019_Parte1/ILE_Reporte_Final.html
