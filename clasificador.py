import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, labels):
        self.centroids = np.array([np.zeros(36) for _ in range(4)])#Initialization is static for convenience, can be made dynamic if necesary
        self.labels = labels.unique()
        self.class_mapping = {cls: i for i, cls in enumerate(self.labels)} #Dictionary to make logging confusion matrix easier
    
    def fit(self, data_X):
        npdata = data_X.to_numpy()
        for i in range(self.labels.shape[0]):
            for j in range(int(np.shape(npdata)[0]/4)*i,int(np.shape(npdata)[0]/4)*(i+1)): #Assumes data is ordered and in the same format used
                self.centroids[i] = np.add(self.centroids[i], npdata[j])
            self.centroids[i] = self.centroids[i]/int(np.shape(npdata)[0]/4)
    def predict(self,  data_X, data_Y):
        errors = 0      #Intialize performance metrics
        c_Matrix = np.zeros([self.labels.shape[0],self.labels.shape[0]])

        npdata = data_X.to_numpy()
        for i in range(np.shape(npdata)[0]):
            #Calculate closest centroid
            distances = np.linalg.norm(npdata[i] - self.centroids, axis=1)
            cluster = np.argmin(distances)
            predicted_label = self.labels[cluster]
            real_label = data_Y.iloc[i]


            #Log results
            c_Matrix[self.class_mapping[predicted_label],self.class_mapping[real_label]] += 1

            if predicted_label != real_label:
                errors += 1
        #Show resuts
        performance = errors/np.shape(npdata)[0]
        print(f"Accuracy {performance*100}%")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(c_Matrix, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(c_Matrix.shape[1]),
           yticks=np.arange(c_Matrix.shape[0]),
           xticklabels=self.labels, yticklabels=self.labels,
           ylabel='True Label',
           xlabel='Predicted Label')
        plt.xticks(rotation=45)

        # Display the values on the matrix

        thresh = c_Matrix.max() / 2
        for i in range(c_Matrix.shape[0]):
            for j in range(c_Matrix.shape[1]):
                ax.text(j, i, format(c_Matrix[i, j]),
                        ha="center", va="center",
                        color="white" if c_Matrix[i, j] > thresh else "black")

        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()



        
        
if __name__ == '__main__':
    df = pd.read_csv("features_with_classes.csv")
    x = df.drop(columns='class')
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    model = Classifier(y_train)
    model.fit(X_train)
    model.predict(X_test,y_train)
    
