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
# 3.A
pca = PCA(n_components=4)
componentes_principales = pca.fit_transform(datos_escalados)
varianza_explicada = pca.explained_variance_ratio_

#---------------------------------------------------------------------

print("\n## Varianza Explicada por PCA (Acumulada)")
varianza_cumulativa = np.cumsum(varianza_explicada)
print(f"CP1: {varianza_explicada[0]:.3f}")
print(f"CP1 + CP2: {varianza_cumulativa[1]:.3f}") 

# Si queremos conservar un 80% de varianza, 2 componentes son insuficientes 
# (solo cubren el 60%), pero son útiles para visualización preliminar.

# --- t-SNE para Visualización ---
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(datos_escalados)
tsne_df = pd.DataFrame(data_tsne, columns=['TSNE1', 'TSNE2'])

# 3.B
# Rango de K a probar, incluyendo los valores sugeridos.
range_k = [3, 4, 5, 15] + list(range(2, 10))
range_k = sorted(list(set(range_k))) # Evitar duplicados

silhouette_scores = {}

for k in range_k:
    if k >= len(data): continue
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(datos_escalados)
    score = silhouette_score(datos_escalados, clusters)
    silhouette_scores[k] = score

# Encontrar el K con la mayor puntuación de Silueta
best_k = max(silhouette_scores, key=silhouette_scores.get)
best_score = silhouette_scores[best_k]

print("\n## Puntuaciones de Silueta para K-Means")
for k, score in silhouette_scores.items():
    print(f"K={k}: {score:.4f}")

print(f"\n--- EL MEJOR K EN EL RANGO ES K={best_k} con Puntuación de Silueta: {best_score:.4f} ---")

# 3.C (Ajuste del Modelo Final)
# Entrenar el modelo final con K=3 porque según la métrica de la silueta, el mejor desempeño se obtiene con K = 3
final_k = 3
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
clusters_final = kmeans_final.fit_predict(datos_escalados)

# Añadir los clústeres a los datos originales para el perfilado
df['Cluster'] = clusters_final

#4
# Perfilado de clústeres usando los datos originales (no escalados)
cluster_profiles = df.groupby('Cluster')[data.columns].mean()

print("\n## Perfilado de Clústeres (Media de Sensores Originales)")
print(cluster_profiles.T.to_markdown(numalign="left", stralign="left"))

# Visualización de los clústeres en 2D (t-SNE)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=clusters_final, cmap='viridis', s=50, alpha=0.6)
plt.title(f'Agrupación K-Means (K={final_k}) proyectada con t-SNE')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.colorbar(scatter, ticks=range(final_k), label='Número de Clúster')
plt.grid(True, alpha=0.2)
plt.show()
