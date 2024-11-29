import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

file_path = "features_with_classes.csv"
data = pd.read_csv(file_path)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

pca = PCA()
x_pca = pca.fit_transform(x)

cumulative_variance = pca.explained_variance_ratio_.cumsum()
n_components = (cumulative_variance >= 0.96).argmax() + 1

print(f"Number of components to explain 96% of variance: {n_components}")

pca_final = PCA(n_components=n_components)
x_pca_final = pca_final.fit_transform(x)

tsne = TSNE(n_components=n_components)
x_tsne = tsne.fit_transform(x)

columns = [f"PCA_{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(data=x_pca_final, columns=columns)
pca_df["class"] = y


columns = [f"TSNE_{i+1}" for i in range(n_components)]
tsne_df = pd.DataFrame(data=x_tsne, columns=columns)
tsne_df["class"] = y

output_csv = "pca_96_features.csv"
pca_df.to_csv(output_csv, index=False)
print(f"PCA features saved to {output_csv}")

output_csv = "tsne_96_features.csv"
tsne_df.to_csv(output_csv, index=False)

print(f"TSNE features saved to {output_csv}")
