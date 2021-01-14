# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 21:29:58 2021

@author: KUDZO VENUNYE ADZAWLA
"""

#------------------------------------------------------------------------------
#
#                                   churn modeling
#
#------------------------------------------------------------------------------

# 1- DATA PREPROCESSING

# importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import keras

# Setting the working directory
os.getcwd()

os.chdir("C:\\Users\\PC\\Desktop\\TO DO LIST\\CUSTOMER CHURN_SUPERDATASCIENCE")

# Importing the dataset

df=pd.read_csv("Churn_Modelling.csv")
x=df.iloc[:,3:13].values
y=df.iloc[:,13].values

### Encoding categorical data. Because of these variables are categorical we need
# to transform them in numeric

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# For X
labelencoder_X_1 = LabelEncoder()
x[:,1] = labelencoder_X_1.fit_transform(x[:,1])

# For Y
labelencoder_X_2 = LabelEncoder()
x[:,2] = labelencoder_X_2.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]


## This is to round to 2 digits
np.set_printoptions(precision=2)


## This is to Suppress scientific notation
np.set_printoptions(suppress=True)



### Spliting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


######### Feature scaling.
# It is vary important in Neural Network Implementation
# Because there are a lot of parallel computation to do do.So it is important 
# to scale in other to ease the calculation

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


####### PART 2 : MAKING THE ANN
# Importing keras package and modules
import keras
from keras.models import Sequential # Initializing the ANN
from keras.layers import Dense # This to create the net layers

## Initializing the ANN. Here we define the ANN as a sequence of layers.These are the 7 steps for building ANN

# STEP 1: Randomly initialise the weights to small numbers close to 0 (but not 0 ).
# STEP 2: Input the first observation of your dataset in the input layer, each feature in one input node.
# STEP 3: Forward-Propagation: from left to right, the neurons are activated in a way that the impact of 
# each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result
# STEP 4: Compare the predicted result to the actual result. Measure the generated error.
# STEP 5: Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how much 
# they are responsible for the error. The learning rate decides by how much we update the weights.
# STEP 6: Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning). Or:
# Repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch Learning).
# STEP 7: When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

# for our NN, the number of nods is equal to the number of independent variables (11 in our case)
classifier = Sequential()

# Adding the input layer and the first hidden layer
# There is no rule of thumbs in choosing the optimum number of nods in the hiden layers.
# But there are tips among which we consider the average of the number of input in the entry layer and the number of input in the output layer.
# In our case this 6. 11 for entry layers and 1 for output layer. their sum divided by 2.
classifier.add(Dense(output_dim=6, init="uniform", activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init="uniform", activation='relu'))

# Adding the Output layer
#classifier.add(Dense(output_dim=1, kernel_initializer='uniform', activation="sigmoid"))
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

# Finting the ANN to the training set
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

# Predicting the test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
