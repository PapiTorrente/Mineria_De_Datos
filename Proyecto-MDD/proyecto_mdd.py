# Considere una muestra inicial con los datos proporcionados en el archivo 'muestra4s.csv'.
# Implemente y ajuste modelos de Aprendizaje Automático No Supervisado de Agrupación, Asociación,
# Reducción de Dimensionalidad u Otro.
# ENCUENTRE UN DESEMPEÑO RAZONABLE.

# Se sabe que son 3, 4, 5 ó 15 k grupos (no más), asuma que lo sabe o no.
# Es decir, puede ocupar modelos que requieren conocer el "k" número de grupos o modelos que buscan ese número sin
# conocimiento a priori de k.

# Import para la manipulación de datos y lectura de archivos (csv en este caso).
import pandas as pd
# Import para escalar los datos y clustering.
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Import para reducción de dimensiones
from sklearn.decomposition import PCA
# Import para 
from sklearn.manifold import TSNE

# LEER EL ARCHIVO
print("\nLectura del archivo.")
datos = pd.read_csv('muestra4s.csv')

if datos is None:
    print("Error en la lectura del archivo.")
else:
    print("Lectura de datos correcto.")
    # Imprime cuántos datos hay en el archivo y cuáles.
    print("Hay ", datos.shape, " datos de la dupla filas-columnas.")
    print("\n\nCabeza del archivo.")
    print(datos.head())

print("\nResúmen Estadístico")
print(datos.describe())

# ESTANDARIZACIÓN Y ESCALADO
# Necesario para la reducción de dimencionalidad con PCA para obtener la media cero
# (Mu = 0) y su desviación estándar (sigma = 1).
escalador = StandardScaler()
datos_escalados = escalador.fit_transform(datos)

print("\nCabeza de los datos escalados:")
print(pd.DataFrame(datos_escalados, columns=datos.columns).head())

# Modelos que requieren conocer K y reducción de Dimensionalidad
# 4
pca = PCA(n_components=4)
componentes_principales = pca.fit_transform(datos_escalados)
varianza_e = pca.explained_variance_ratio_
