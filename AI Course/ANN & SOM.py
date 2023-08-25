# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:30:25 2023

@author: TGerb
"""

# Part 1 - Identify the Frauds with the Self-Organizing Map
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling (Skallierung anpassen durch z.B. Normalisierung)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # Normalisierung nicht Standartisierung
X = sc.fit_transform(X)


# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone() # der Rahmen
pcolor(som.distance_map().T) #Transposition der Matrix. Hier wird die Map des SOMs erstellt
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x) #we get the first node for the winning customer. at the start i will be zero and x will be the first element of X, and that is the first customer
    plot(w[0] + 0.5, #so for this winning node "w", we will place the colored marker on it
         w[1] + 0.5,
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()    

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(3,1)], mappings[(5,5)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# Part 2 - Going from Unsupervised to Supervised Deep Learning
# Creating the matrix of features
customers  = dataset.iloc[:, 1:].values # [welche zeilen , welche spalten]

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling / Normierung
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Building the ANN
### Initializing the ANN
# Importing Keras Libaries 
from keras.models import Sequential
from keras.layers import Dense

# Initializse the ANN
classifier = Sequential()

# Adding input layer and hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Make the predicitons / Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
















