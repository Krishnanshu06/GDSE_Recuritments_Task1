import tensorflow as tf
from keras import datasets , layers , models
import numpy as np

(X_train , y_train) , (X_test ,y_test) = datasets.cifar10.load_data()

# print(X_train.shape)
# print(X_train[0].shape)
# print(X_train[0])
# print(y_train[:5])

outputClasses = ["airplane" , "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

y_train = y_train.reshape(-1,) #making it a 1d array
print(y_train[:5])


X_train = X_train / 255
X_test = X_test / 255        #normalizing the data to be btw 0 and 1

print(X_train[0])
