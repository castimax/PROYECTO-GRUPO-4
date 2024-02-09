# Análisis de Modelos de Aprendizaje Supervisado y Resultados

En esta sección, implementamos y evaluamos tres modelos de aprendizaje supervisado: Regresión Logística, Máquinas de Vectores de Soporte (SVM) y Árboles de Decisión. A continuación, presentamos una descripción general de cada método y los resultados obtenidos en el análisis del conjunto de datos de casos y genes.

---

## Regresión Logística

La regresión logística es un método de clasificación que modela la probabilidad de pertenecer a una clase utilizando la función logística. Este método es ampliamente utilizado para problemas de clasificación binaria y multiclase. En nuestro análisis, implementamos un modelo de regresión logística y evaluamos su desempeño en la clasificación de casos basada en los genes observados.

### Resultados de Regresión Logística

El modelo de regresión logística alcanzó los siguientes resultados en el conjunto de datos:

- Matriz de confusión:
|  | Predicción 1 | Predicción 2 | Predicción 3 | Predicción 4 | Predicción 5 |
| --- | --- | --- | --- | --- | --- |
| Clase 1 | 43 | 0 | 0 | 0 | 0 |
| Clase 2 | 0 | 98 | 0 | 0 | 0 |
| Clase 3 | 0 | 0 | 40 | 0 | 0 |
| Clase 4 | 0 | 0 | 0 | 40 | 0 |
| Clase 5 | 0 | 0 | 0 | 0 | 19 |

- Precisión: 1.0
- Sensibilidad: 1.0
- Especificidad: 1.0
- Puntuación F1: 1.0

## Máquinas de Vectores de Soporte (SVM)

Las Máquinas de Vectores de Soporte son un método de clasificación que encuentra el hiperplano óptimo que separa las clases en el espacio de características. SVM es efectivo en la clasificación de datos lineal y no lineal. En nuestro análisis, implementamos un clasificador SVM para predecir las clases de casos basadas en la expresión génica.

### Resultados de SVM

El clasificador SVM obtuvo los siguientes resultados en el conjunto de datos:

- Matriz de confusión:
|  | Predicción 1 | Predicción 2 | Predicción 3 | Predicción 4 | Predicción 5 |
| --- | --- | --- | --- | --- | --- |
| Clase 1 | 43 | 0 | 0 | 0 | 0 |
| Clase 2 | 0 | 98 | 0 | 0 | 0 |
| Clase 3 | 0 | 0 | 40 | 0 | 0 |
| Clase 4 | 0 | 0 | 0 | 40 | 0 |
| Clase 5 | 0 | 0 | 0 | 0 | 19 |

- Precisión: 1.0
- Sensibilidad: 1.0
- Especificidad: 1.0
- Puntuación F1: 1.0

## Árboles de Decisión

Los Árboles de Decisión son estructuras de árbol que dividen el conjunto de datos en subconjuntos más pequeños basados en características específicas. Estos modelos son útiles para problemas de clasificación y pueden ser fácilmente interpretados. En nuestro análisis, construimos un árbol de decisión para clasificar los casos basados en la expresión génica.

### Resultados de Árboles de Decisión

El árbol de decisión produjo los siguientes resultados en el conjunto de datos:

- Matriz de confusión:

|  | Predicción 1 | Predicción 2 | Predicción 3 | Predicción 4 | Predicción 5 |
| --- | --- | --- | --- | --- | --- |
| Clase 1 | 42 | 1 | 0 | 0 | 0 |
| Clase 2 | 0 | 94 | 2 | 2 | 0 |
| Clase 3 | 0 | 0 | 40 | 0 | 0 |
| Clase 4 | 0 | 0 | 0 | 40 | 0 |
| Clase 5 | 0 | 0 | 0 | 0 | 19 |

- Precisión: 0.9798
- Sensibilidad: 0.9792
- Especificidad: 0.9792
- Puntuación F1: 0.9792
