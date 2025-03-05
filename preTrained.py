import tensorflow as tf
from keras import datasets , layers , models
import numpy as np


#############################################################################################################################
#initializing and normalizing data

(X_train , y_train) , (X_test ,y_test) = datasets.cifar10.load_data()

outputClasses = ["airplane" , "automobile" , "bird" , "cat" , "deer" , "dog" , "frog" , "horse" , "ship" , "truck"]

y_train = y_train.reshape(-1,) #making it a 1d array

X_train = X_train / 255
X_test = X_test / 255        #normalizing the data to be btw 0 and 1


#############################################################################################################################
# Importing VGG16

VGG_layers = tf.keras.applications.VGG16(weights = 'imagenet',
                                         include_top = False,   
                                         input_shape = (32,32,3))

VGG_layers.trainable = False   # make the Conv layers untrainable

for layer in VGG_layers.layers[-4:]:
    layer.trainable = True


#############################################################################################################################
# Hyperparameters


dropoutProb4 = 0.4
batchSize = 100
nEpoch = 30
InitLR = 0.0005



#############################################################################################################################
#Model

CnnModel = models.Sequential([

                VGG_layers,
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(dropoutProb4),
                layers.Dense(256, activation='relu'),
                layers.Dropout(dropoutProb4),
                layers.Dense(10, activation='softmax')

                ])


optimizerAdam = tf.keras.optimizers.Adam(learning_rate = InitLR)

CnnModel.compile(optimizer=optimizerAdam, 
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
# this is the callback for the early stopping regularization meathod.
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
# reducing lr when the loss is stagnent to converge faster.


#############################################################################################################################
#Training


SaveHistory=CnnModel.fit(X_train , y_train , 
                        epochs= nEpoch,
                        validation_data=(X_test,y_test),
                        batch_size= batchSize,
                        callbacks=[early_stopping,lr_scheduler])


#############################################################################################################################
# Evaluation metrics

from sklearn.metrics import classification_report

predictions = np.argmax(CnnModel.predict(X_test) , axis=1)
true = y_test.reshape(-1,)

print(classification_report(true , predictions, target_names= outputClasses))


#############################################################################################################################
#plotting acc and loss per epoch

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# acc plot
plt.subplot(1, 2, 1)
plt.plot(SaveHistory.history['accuracy'], label='Train Accuracy')
plt.plot(SaveHistory.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')

# for loss plot
plt.subplot(1, 2, 2)
plt.plot(SaveHistory.history['loss'], label='Train Loss')
plt.plot(SaveHistory.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')

plt.show()


# even after i tune the hyperparameters, The maximum validation accuracy is about 78%.
# for further improvements we will use a pre trianed model and build our model on it.