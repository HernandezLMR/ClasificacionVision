import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

file_path = "features_with_classes.csv"
data = pd.read_csv(file_path)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(x)

tsne_df = pd.DataFrame(data=x_tsne, columns=["TSNE1", "TSNE2"])
tsne_df["class"] = y

plt.figure(figsize=(10, 10))
for class_name in tsne_df["class"].unique():
    class_data = tsne_df[tsne_df["class"] == class_name]
    plt.scatter(class_data["TSNE1"], class_data["TSNE2"], label=class_name)
plt.legend()
plt.title("TSNE Features")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.grid()
plt.show()

output_csv = "tsne_features.csv"
tsne_df.to_csv(output_csv, index=False)

print(f"TSNE features saved to {output_csv}")
