from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))

def cargar_datos(archivo):
    return pd.read_csv(os.path.join(DIRECTORIO, "DATOS", archivo), sep=";")

# Cargar nombres de columnas
with open(os.path.join(DIRECTORIO, "DATOS", "nombres_columnas.txt")) as file:
    col_nombres = [line.strip() for line in file]

# Cargar datos
filas_clases = cargar_datos("filas_clases.csv")
expresion_genes = cargar_datos("tabla_gene_expression.csv")

# Asegurarse de que la primera columna se llame 'Clases' y ajustar los nombres de las columnas
col_nombres_ajustados = ['Clases'] + col_nombres

# Crear DataFrame con nombres de columnas ajustados y la longitud adecuada
df = pd.DataFrame(index=expresion_genes.index, columns=col_nombres_ajustados[1:len(col_nombres)+1])

# Llenar el DataFrame con los datos de expresión de genes
df.iloc[:, 0:len(col_nombres)] = expresion_genes.iloc[:, :len(col_nombres)].values

# Asignar Clases
df.insert(0, 'Clases', filas_clases.iloc[:, 1].values)

# Renombrar índices para que coincidan con los sample_id
df.index = ['sample_' + str(i) for i in range(len(df))]

# Limpiar datos excluyendo Clases (suponiendo que todas las demás columnas son numéricas)
cols_datos = df.columns[1:]
df[cols_datos] = df[cols_datos].apply(pd.to_numeric, errors='coerce').round(2).fillna(df[cols_datos].mean())

# Seleccionar las primeras 10 filas y las primeras 10 columnas para mostrar
df_small = df.iloc[:10, :11]
# Mostrar la tabla pequeña
print(df_small)

# IMPLEMENTACION DE KNN Y ESTANDARIZAMOS DATOS DEL DF

# Separar la columna 'Clases' y los datos para imputación
clases = df['Clases']
datos_para_imputar = df.drop('Clases', axis=1)

# Crear y aplicar KNNImputer
imputer = KNNImputer(n_neighbors=5)
datos_imputados = imputer.fit_transform(datos_para_imputar)

# Crear y aplicar StandardScaler
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_imputados)

# Reconvertir a DataFrame y reinsertar la columna 'Clases'
df_escalado = pd.DataFrame(datos_escalados, columns=datos_para_imputar.columns, index=df.index)
df_escalado.insert(0, 'Clases', clases)

# Seleccionar las primeras 10 filas y las primeras 10 columnas para mostrar
df_escalado_small = df_escalado.iloc[:10, :11]

# Mostrar la tabla pequeña con estandarización e imputación aplicadas
print("Tabla con Imputación KNN y Estandarización aplicadas:")
print(df_escalado_small)
# Análisis de la Tabla Estándarizada y la Imputación con KNN
explicacion = """
## Análisis de la Tabla Estándarizada y la Imputación con KNN

Hemos aplicado la estandarización a nuestro conjunto de datos para transformar las variables numéricas,
de modo que tengan una media de 0 y una desviación estándar de 1. Al observar la segunda tabla, donde los 
valores han sido estandarizados, podemos ver que:

1. Los valores negativos indican puntuaciones por debajo de la media del conjunto de datos. Por ejemplo,
   una puntuación z de -1.0 significa que la expresión del gen está una desviación estándar por debajo de la media de la muestra.
2. Esta transformación facilita la comparación directa entre distintos genes y nos permite preparar el conjunto de datos
   para algoritmos de aprendizaje automático que requieren que todas las características estén en una escala común.
"""

print(explicacion)

# Métodos de reducción de dimensionalidad y clusterizacion

explicacion0 = """
## Introducción al Análisis de Modelos de Aprendizaje No Supervisado y Reducción de Dimensionalidad ##

En esta sección, exploramos la estructura inherente en nuestros datos utilizando técnicas de aprendizaje no supervisado y reducción de dimensionalidad. Empleamos cuatro métodos diferentes para comprender mejor la distribución de nuestros datos y encontrar patrones significativos. A continuación, presentamos una visión general de los métodos utilizados y los resultados obtenidos:

### Reducción de Dimensionalidad ###

Para reducir la dimensionalidad de nuestros datos y visualizar su estructura de manera más efectiva, aplicamos dos técnicas principales:

1. **Análisis de Componentes Principales (PCA):** Esta técnica nos permite reducir la dimensionalidad de nuestros datos mientras mantenemos la mayor cantidad posible de información. PCA transforma las características originales en un conjunto de componentes principales no correlacionados, lo que facilita la visualización de la estructura de nuestros datos en un espacio de menor dimensión.

2. **t-Distributed Stochastic Neighbor Embedding (t-SNE):** t-SNE es una técnica de reducción de dimensionalidad no lineal que se utiliza principalmente para la visualización de datos de alta dimensión. Se enfoca en preservar la estructura local de los datos, lo que lo hace especialmente útil para descubrir patrones intrínsecos y relaciones no lineales en los datos.

### Clusterización ###

Después de reducir la dimensionalidad de nuestros datos, aplicamos dos algoritmos de clusterización para identificar grupos naturales o patrones dentro de nuestros datos:

1. **K-Means:** Utilizamos el algoritmo K-Means para agrupar los datos en k grupos basados en la similitud de las características. Este método es útil para identificar grupos compactos y bien separados en nuestros datos.

2. **Clustering Jerárquico Aglomerativo:** Este método de clusterización construye un árbol jerárquico de grupos mediante la fusión de pares de grupos similares. Nos permite explorar la estructura jerárquica de nuestros datos y identificar grupos a diferentes niveles de similitud.

### Resultados y Objetivos del Análisis ###

Los resultados obtenidos de estas técnicas nos permiten comprender mejor la estructura y los patrones presentes en nuestros datos. Nuestro objetivo principal es utilizar esta comprensión para tomar decisiones informadas y desarrollar estrategias efectivas para abordar nuestro problema específico.

"""
print(explicacion0)

# PCA
pca = PCA(n_components=2)  # Reducir a 2 dimensiones
pca_result = pca.fit_transform(df_escalado.drop('Clases', axis=1))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000)
tsne_result = tsne.fit_transform(df_escalado.drop('Clases', axis=1))

# Métodos de clustering
# K-Means
kmeans = KMeans(n_clusters=3)  # Suponiendo que queremos dividir en 3 clústeres
kmeans_clusters = kmeans.fit_predict(df_escalado.drop('Clases', axis=1))

# Clustering Jerárquico Aglomerativo
agglomerative = AgglomerativeClustering(n_clusters=3)
agglo_clusters = agglomerative.fit_predict(df_escalado.drop('Clases', axis=1))

# Ahora podrías agregar los resultados como columnas adicionales al DataFrame o utilizarlos para visualizar o analizar más
df_escalado['PCA1'] = pca_result[:, 0]
df_escalado['PCA2'] = pca_result[:, 1]
df_escalado['tSNE1'] = tsne_result[:, 0]
df_escalado['tSNE2'] = tsne_result[:, 1]
df_escalado['KMeans_Cluster'] = kmeans_clusters
df_escalado['Agglo_Cluster'] = agglo_clusters

#VER LOS RESULTADOS EN TEXTO Y CON DATOS

# Imprimir resultados PCA
print("Resultados PCA:")
print(df_escalado[['PCA1', 'PCA2']].head())

# Imprimir resultados t-SNE
print("Resultados t-SNE:")
print(df_escalado[['tSNE1', 'tSNE2']].head())

# Imprimir asignaciones de clúster K-Means
print("Asignaciones de clúster K-Means:")
print(df_escalado['KMeans_Cluster'].head())

# Imprimir asignaciones de clúster Jerárquico Aglomerativo
print("Asignaciones de clúster Jerárquico Aglomerativo:")
print(df_escalado['Agglo_Cluster'].head())


# VER RESULTADO GRAFICAMENTE (veo que no sale por terminal de spacecode de Github)
#prueba de mostrar graficos

# Función para plotear resultados
def plot_results(features, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Plotear PCA
plot_results(pca_result, df_escalado['KMeans_Cluster'], "PCA Resultados con Clusters K-Means")
# Guardar el gráfico como un archivo de imagen así los muestro en md y/o html
plt.savefig('mi_PCA.png')


# Plotear t-SNE
plot_results(tsne_result, df_escalado['KMeans_Cluster'], "t-SNE Resultados con Clusters K-Means")
# Guardar el gráfico como un archivo de imagen
plt.savefig('mi_t-SNE.png')


# Plotear Clústeres K-Means
plot_results(pca_result, df_escalado['KMeans_Cluster'], "Clústeres K-Means en PCA reducido")
# Guardar el gráfico como un archivo de imagen
plt.savefig('mi_PCAreduc.png')

# Plotear Clústeres Jerárquicos Aglomerativos
plot_results(pca_result, df_escalado['Agglo_Cluster'], "Clústeres Jerárquicos Aglomerativos en PCA 2 reducido")
# Guardar el gráfico como un archivo de imagen
plt.savefig('mi_PCA2reduc.png')

#AHORA LOS ANALISIS SUPERVISADOS
explicacion2 = """

## Análisis de Modelos de Aprendizaje Supervisado ##

En esta sección, llevamos a cabo la implementación y evaluación de tres modelos de aprendizaje supervisado: Regresión Logística, Máquinas de Vectores de Soporte (SVM) y Árboles de Decisión.

### Regresión Logística ###

La regresión logística es un método de clasificación que utiliza la función logística para modelar la probabilidad de una clase binaria o multiclase. Entrenamos un modelo de regresión logística utilizando nuestros datos y evaluamos su desempeño utilizando varias métricas de evaluación.

### Máquinas de Vectores de Soporte (SVM) ###

Las Máquinas de Vectores de Soporte son un método de clasificación que busca encontrar el hiperplano óptimo que mejor separa las clases en el espacio de características. Implementamos un clasificador SVM y evaluamos su rendimiento en nuestro conjunto de datos.

### Árboles de Decisión ###

Los Árboles de Decisión son un método de aprendizaje supervisado que divide repetidamente el conjunto de datos en subconjuntos más pequeños basados en características específicas, con el objetivo de clasificar las muestras en clases homogéneas. Construimos un árbol de decisión y evaluamos su capacidad para predecir las clases de nuestro conjunto de datos.

"""

print(explicacion2)


print
# Asumiendo que 'df_escalado' es tu DataFrame con las características estandarizadas y 'Clases' es la etiqueta
X = df_escalado.drop('Clases', axis=1).values
y = df_escalado['Clases'].values

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar los clasificadores
models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

# Entrenar cada modelo y calcular métricas
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    # Imprimir las métricas
    print(f"Modelo: {name}")
    print("Matriz de confusión:")
    print(conf_matrix)
    print(f"Precisión: {precision}")
    print(f"Sensibilidad: {recall}")
    print(f"Especificidad: {accuracy}")  # La especificidad se debe calcular de forma diferente para multiclase
    print(f"Puntuación F1: {f1}")
    print("\n")