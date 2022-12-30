# -*- coding: utf-8 -*-
"""Multiclass_image_ANN.ipynb

@author: Bmadios
@Date: 2022-12-29

# Step 1: Installation and Setup
"""

# Installing Tensorflow
#!pip install tensorflow-gpu

import tensorflow as tf
print(tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""# Step 2: Data preprocessing

"""

# Loading dataset
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train.shape, y_train.shape, X_test.shape

np.max(y_train), np.min(y_train) # 10 Classes 0 to 9

class_names = ["0 Top/Tshirt", "1 Trouser", "2 Pullover", "3 Dress", "4 Coat", "5 Sandal", "6 Shirt", "7 Sneaker", "8 Bag", "9 Ankle boot"]

# Data exploration
plt.figure()
plt.imshow(X_train[1])
plt.colorbar()

# Normalize data
X_train, X_test = X_train/255, X_test/255
plt.figure()
plt.imshow(X_train[1])
plt.colorbar()
# Reduce data dimensions 3D to 1D
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)



"""# Step 3: Building the model"""

model  = tf.keras.models.Sequential()
# Add the input layer
# 1/ Units (No of neurons) = 128
# 2/ Activation function: ReLU
# 3/ Input shape = (784,) 
model.add(tf.keras.layers.Dense(units=140, activation="relu", input_shape=(784,)))
# Add the hidden layer to prevent Overfitting
model.add(tf.keras.layers.Dropout(0.3))
# Add the output layer
# 1/ Units = 10
# 2/ Activation function: softmax
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

"""# Step 4: Training the model"""

# Compile the model with following parameters
# 1. Optimizer = adam (minimize the loss function)
# 2. loss function = sparse_categorical_crossentropy (acts as guide to optimizer)
# 3. metrics = sparse_categorical_accuracy

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="sparse_categorical_accuracy")
model.summary()

model.fit(X_train, y_train, epochs=20)

"""# Step 5: Model performance and Prediction"""

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Accuracy is:{:.2f} %".format(test_accuracy*100))

predict_x=model.predict(X_test) 
y_pred=np.argmax(predict_x,axis=1)
y_pred[100], y_test[100] # Predicted and actual class value

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
conf_mat = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = [False, True])
cm_display.plot()
plt.show()