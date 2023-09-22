# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:03:18 2023

@author: TGerb
"""

# Importing the libaries 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn #module of torch to implement neural networks
import torch.nn.parallel #this is for parallel computation
import torch.optim as optim #this is for the optimizer 
import torch.utils.data #this is for the tools used in here
from torch.autograd import Variable #this is for stochastic gradient descent

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the trainingset and the testset
training_set = pd.read_csv('ml-100k/u1.base', sep='\t')
training_set_a = np.array(training_set, dtype='int') #create an array for torch
test_set = pd.read_csv('ml-100k/u1.test', sep='\t')
test_set_a = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set_a[:,0]), max(test_set_a[:,0])))
nb_movies = int(max(max(training_set_a[:,1]), max(test_set_a[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings)) #basicly a list of lists
    return new_data

training_set_b = convert(training_set_a)
test_set_b = convert(test_set_a)

# Converting the data (the lists inside a list) into Torch tensors
training_set_pytorchtensor = torch.FloatTensor(training_set_b)
test_set_pytorchtensor = torch.FloatTensor(test_set_b)

# Converting the ratings into binary rating 1 (Liked) or 0 (Not liked)
# From here we start with the specific stuff for BoltzmanMachines
training_set_pytorchtensor[training_set_pytorchtensor == 0]  = -1 # Take all "0" inside the tensor and change them
training_set_pytorchtensor[training_set_pytorchtensor == 1]  = 0 # Or Operator doesnt work with PyTorch like with Python
training_set_pytorchtensor[training_set_pytorchtensor == 2]  = 0
training_set_pytorchtensor[training_set_pytorchtensor >= 3]  = 1

test_set_pytorchtensor[test_set_pytorchtensor == 0]  = -1 # Take all "0" inside the tensor and change them
test_set_pytorchtensor[test_set_pytorchtensor == 1]  = 0 # Or Operator doesnt work with PyTorch like with Python
test_set_pytorchtensor[test_set_pytorchtensor == 2]  = 0
test_set_pytorchtensor[test_set_pytorchtensor >= 3]  = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh): #visible node, hidden node
        self.W = torch.randn(nh, nv) # Weights.
        self.a = torch.randn(1, nh) # Bias. 2D Tensor needed since 1-D Tensor doesnt work. 1 corresponding to the first dimension is the batch.
                                    # There is a Bias for each hidden node and we have nh hidden nodes
        self.b = torch.randn(1, nv) # Bias for visible Nodes. Bias tells how the value has to be for an active node if i understand correctly
    
    def sample_h(self, x): # use self to access the variable in __init__, x correspeond to visible neurons in p(h|v)
        # First compute the probability of p(h|v), "probability h given v" -> that is basicly the sigmoid activation fuction
        wx = torch.mm(x, self.W.t()) # .mm to create a product of 2 tensors
        activation = wx + self.a.expand_as(wx) # now we compute what is inside the acitvation function
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy) 
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk): 
        '''
        v = input vector (containing the ratings of all movies by one user)
        vk = the visible nodes obtained after K samplings. 
        ph0 =  that's the vector of probabilities that at the first iteration the hidden nodes equal one given the values of v0
        phk = the probabilitie of the hidden nodes after K sampling given the values of the visible nodes, VK.
        '''
        #print("torch.mm(v0.t(), ph0) shape:", torch.mm(v0.t(), ph0).shape)
        #print("torch.mm(vk.t(), phk) shape:", torch.mm(vk.t(), phk).shape)
        #print("self.W shape:", self.W.shape)

        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set_pytorchtensor[0]) 
nh = 100
batch_size = 100 #with 1 its basicly online learning -> updating the network after each observation
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0 
    s = 0. # a counter to normalize the loss function train_loss
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set_pytorchtensor[id_user:id_user+batch_size]
        v0 = training_set_pytorchtensor[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        
        #print("v0 shape: ", v0.shape)
        #print("vk shape: ", vk.shape)
        #print("ph0 shape: ", ph0.shape)
        #print("phk shape: ", phk.shape)
        
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1. 
        print('Epoch: ' +str(epoch)+' loss: '+str(train_loss/s))
        
# Testing the RBM

test_loss = 0 
s = 0. 
for id_user in range(0, nb_users): # now looping over all users one by one
    v = training_set_pytorchtensor[id_user:id_user+1]
    vt = test_set_pytorchtensor[id_user:id_user+1] #vt = visible target
    if len(vt[vt>=0]) > 0:     
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
    s += 1. 
print('test loss: '+str(test_loss/s))
        


















































