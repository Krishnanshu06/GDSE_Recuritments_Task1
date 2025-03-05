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
# print(y_train[:5])




X_train = X_train / 255
X_test = X_test / 255        #normalizing the data to be btw 0 and 1

# print(X_train[0])


dropoutProb1 = 0.2
dropoutProb2 = 0.3
dropoutProb3 = 0.4
dropoutProb4 = 0.5
batchSize = 100
nEpoch = 30




CnnModel = models.Sequential([

                layers.Conv2D(filters=32 , kernel_size =(3,3) , activation= 'relu' , input_shape= X_train[0].shape),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2,2)),
                layers.Dropout(dropoutProb1),

                layers.Conv2D(filters=64 , kernel_size= (3,3) , activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2,2)),
                layers.Dropout(dropoutProb2),

                layers.Conv2D(filters=128 , kernel_size= (3,3) , activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2,2)),
                layers.Dropout(dropoutProb3),

                layers.Flatten(),
                layers.Dense(256 , activation='relu'),
                layers.Dropout(dropoutProb4),
                layers.Dense(10 , activation= 'softmax')

                ])


CnnModel.compile(optimizer='adam', 
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# this is the callback for the early stopping regularization meathod.


SaveHistory=CnnModel.fit(X_train , y_train , 
                        epochs= nEpoch,
                        validation_data=(X_test,y_test),
                        batch_size= batchSize,
                        callbacks=[early_stopping])

predictions = np.argmax(CnnModel.predict(X_test) , axis=1)
true = y_test.reshape(-1,)

from sklearn.metrics import classification_report

print(classification_report(true , predictions, target_names= outputClasses))



