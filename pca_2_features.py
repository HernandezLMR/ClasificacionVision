import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_path = "features_with_classes.csv"
data = pd.read_csv(file_path)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

pca_df = pd.DataFrame(data=x_pca, columns=["PC1", "PC2"])
pca_df["class"] = y

plt.figure(figsize=(10, 10))
for class_name in pca_df["class"].unique():
    class_data = pca_df[pca_df["class"] == class_name]
    plt.scatter(class_data["PC1"], class_data["PC2"], label=class_name)
plt.legend()
plt.title("PCA Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

output_csv = "pca_features.csv"
pca_df.to_csv(output_csv, index=False)

print(f"PCA features saved to {output_csv}")
