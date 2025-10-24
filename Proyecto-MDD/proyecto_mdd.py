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

# LEER EL ARCHIVO
print("Lectura del archivo.")
datos = pd.read_csv('muestra4s.csv')

if datos is None:
    print("Error en la lectura del archivo.")
else:
    print("Lectura de datos correcto.")
    # Imprime cuantos datos hay en el archivo y cuáles.
    print(datos.shape)
    print(datos.head())

# Eliminar ID.
ids = datos['id']
datosnv = datos.drop('id',axis=1)

# ESCALAR LOS DATOS
escalador = StandardScaler()
datos_escalados = escalador.fit_transform(datosnv)

print("Primeras 5 filas de los datos escalados:")
print(pd.DataFrame(datos_escalados, columns=datosnv.columns).head())
print("\nEstadísticas descriptivas de los datos escalados:")
print(pd.DataFrame(datos_escalados).describe())