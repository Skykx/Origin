# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:16:35 2023

@author: TGerb
"""

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
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

