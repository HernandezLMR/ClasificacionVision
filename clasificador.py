import numpy as np
class Classifier:
    def __init__(self, attributes:np.array):
        self.attributes = attributes
        self.centroids = np.zeros(self.attributes.shape)
    #def  predict(self, )
        
if __name__ == '__main__':
    array = np.array([[[[[[[[[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],[[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]]]]]]]]])
    print(np.shape(array))
    print(array[0][0][0][0][0][0][0][0][8])