import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Cargar los datos
file_path = "features_with_classes.csv"
data = pd.read_csv(file_path)

# Separar características y etiquetas
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Ajustar PCA para calcular la varianza explicada
pca = PCA()
pca.fit(X)
cumulative_variance = pca.explained_variance_ratio_.cumsum()
n_components = (cumulative_variance >= 0.96).argmax() + 1

print(f"Number of components to explain 96% of variance: {n_components}")

# Ajustar PCA con al menos 2 componentes para graficar
pca_final = PCA(n_components=max(n_components, 2))  # Asegurar al menos 2 componentes
X_pca_final = pca_final.fit_transform(X)

# Ajustar t-SNE a exactamente 2 componentes para visualización
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Crear DataFrames
columns_pca = [f"PCA_{i+1}" for i in range(2)]  # Forzar nombres de columnas para 2D
pca_df = pd.DataFrame(X_pca_final[:, :2], columns=columns_pca)
pca_df["class"] = y

columns_tsne = ["TSNE_1", "TSNE_2"]
tsne_df = pd.DataFrame(X_tsne, columns=columns_tsne)
tsne_df["class"] = y

# Visualizar PCA y t-SNE
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico PCA
for class_label in pca_df["class"].unique():
    subset = pca_df[pca_df["class"] == class_label]
    axes[0].scatter(subset["PCA_1"], subset["PCA_2"], label=class_label, alpha=0.7)
axes[0].set_title("PCA Visualization")
axes[0].set_xlabel("PCA_1")
axes[0].set_ylabel("PCA_2")
axes[0].legend()

# Gráfico t-SNE
for class_label in tsne_df["class"].unique():
    subset = tsne_df[tsne_df["class"] == class_label]
    axes[1].scatter(subset["TSNE_1"], subset["TSNE_2"], label=class_label, alpha=0.7)
axes[1].set_title("t-SNE Visualization")
axes[1].set_xlabel("TSNE_1")
axes[1].set_ylabel("TSNE_2")
axes[1].legend()

plt.tight_layout()
plt.show()

# Guardar los DataFrames
output_csv_pca = "pca_96_features_visual.csv"
pca_df.to_csv(output_csv_pca, index=False)
print(f"PCA features saved to {output_csv_pca}")

output_csv_tsne = "tsne_96_features_visual.csv"
tsne_df.to_csv(output_csv_tsne, index=False)
print(f"t-SNE features saved to {output_csv_tsne}")
