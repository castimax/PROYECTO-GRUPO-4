import os
import pandas as pd

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))  

def cargar_datos(archivo):
    return pd.read_csv(os.path.join(DIRECTORIO, "DATOS", archivo), sep=";")

# Cargar nombres de columnas
with open(os.path.join(DIRECTORIO, "DATOS", "nombres_columnas.txt")) as file:
    col_nombres = [line.strip() for line in file]

# Cargar datos  
filas_clases = cargar_datos("filas_clases.csv")
expresion_genes = cargar_datos("tabla_gene_expression.csv")

# Verificar que el número de nombres de columnas corresponde al número de columnas de datos + 1 (por la columna de clases)
if len(col_nombres) == expresion_genes.shape[1] + 1:
    # Los nombres de las columnas y los datos coinciden, proceder con la asignación
    expresion_genes.columns = col_nombres[1:]  # Omitimos el primer nombre que es para la columna de clases
    expresion_genes.insert(0, col_nombres[0], filas_clases.iloc[:, 1])  # Insertamos la columna de clases
elif len(col_nombres) < expresion_genes.shape[1] + 1:
    print("Faltan nombres de columnas en el archivo 'nombres_columnas.txt'.")
    # Aquí podrías agregar un nombre de columna faltante o manejar el error como prefieras
elif len(col_nombres) > expresion_genes.shape[1] + 1:
    print("Hay nombres de columnas de más en el archivo 'nombres_columnas.txt'.")
    # Aquí podrías eliminar el nombre de columna extra o manejar el error como prefieras
