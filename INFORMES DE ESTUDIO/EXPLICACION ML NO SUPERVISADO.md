# Interpretación de los Resultados de Aprendizaje No Supervisado

Como grupo de trabajo, hemos aplicado métodos de aprendizaje no supervisado para entender mejor la estructura subyacente de nuestro conjunto de datos genómicos. Los resultados obtenidos son sumamente informativos:

## PCA (Análisis de Componentes Principales)

El PCA nos permitió reducir la dimensionalidad de nuestro conjunto de datos, resaltando las direcciones de máxima varianza. Los clústeres observados en el gráfico de PCA indican diferencias significativas en la expresión genética entre los grupos, lo cual es evidente por la separación espacial de los puntos.

![PCA Resultados con Clusters K-Means](mi_PCA.png)

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

El t-SNE, al conservar la estructura local de los datos, ha revelado agrupaciones naturales que no eran tan evidentes con el PCA. Esto sugiere que ciertas muestras comparten características genéticas más cercanas entre sí en un espacio multidimensional complejo.

![t-SNE Resultados con Clusters K-Means](mi_t-SNE.png)

## Clustering K-Means

Al aplicar K-Means, hemos identificado grupos que se correlacionan con diferencias fenotípicas o tratamientos específicos dentro de nuestras muestras. Los grupos formados por K-Means son coherentes con la variabilidad observada en las dos primeras componentes principales del PCA.

![Clústeres K-Means en PCA reducido](mi_PCAreduc.png)

## Clustering Jerárquico Aglomerativo

Los clústeres jerárquicos aglomerativos, visualizados en el espacio de PCA reducido, destacan una jerarquía de agrupaciones que podrían corresponder a subtipos genéticos distintos dentro de las muestras analizadas.

![Clústeres Jerárquicos Aglomerativos en PCA 2 reducido](mi_PCA2reduc.png)

Estos métodos nos han proporcionado una visión más clara de la complejidad y la heterogeneidad de los datos genómicos, facilitando la identificación de patrones que podrían ser cruciales para investigaciones futuras.


**En la carpeta VIDEOS están los gráficos en formato presentación**
