import tensorflow as tf
from keras import datasets , layers , models
import numpy as np

(X_train , y_train) , (X_test ,y_test) = datasets.cifar10.load_data()

print(X_train.shape)
