import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep="::", header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep="::", header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep="::", header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', sep='\t')
training_set = np.array(training_set, dtype='int64')
test_set = pd.read_csv('ml-100k/u1.test', sep='\t')
test_set = np.array(test_set, dtype='int64')

# Getting the number of users and movies
nb_users =int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# # Converting the data into an array with users in lines and movies in columns
# def convert(data):
#     new_data = []
#     for id_users in range(1, nb_users+1):
#         id_movies = data[:,1][data[:,0] == id_users]
#         id_ratings = data[:,2][data[:,0] == id_users]
#         ratings = np.zeros(nb_movies)
#         ratings[id_movies-1] = id_ratings
#         new_data.append(list(ratings)) #basicly a list of lists
#     return new_data

# converting teh data into an array with users in lines and movies in columns
def convert_to_user_movie_matrix(data, nb_users, nb_movies):
    # Initialisiere eine Liste von Listen mit Nullen
    user_movie_matrix = np.zeros((nb_users, nb_movies))
    for row in data:
        # Die Indizes in der Datei beginnen bei 1, daher subtrahieren wir 1, um bei 0 zu beginnen
        user_id = row[0] - 1
        movie_id = row[1] - 1
        rating = row[2]
        user_movie_matrix[user_id, movie_id] = rating
    return user_movie_matrix.tolist()  # Konvertiere das Numpy-Array zurück in eine Liste von Listen

# Anwendung der Funktion auf den Trainings- und Testdatensatz
training_set = convert_to_user_movie_matrix(training_set, nb_users, nb_movies)
test_set = convert_to_user_movie_matrix(test_set, nb_users, nb_movies)

# Converting the data in to Torch tesnors
training_set = torch.FloatTensor(training_set) 
test_set = torch.FloatTensor(test_set) 

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() #Superclass, that's will make sure we get all the inherited classes and methods of the parent class and then module.
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x)) #encoding nb_movies -> 20
        x = self.activation(self.fc2(x)) #encoding 20 -> 10
        x = self.activation(self.fc3(x)) #decoding 10 -> 20
        x = self.fc4(x)
        return x 
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

#Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = training_set[id_user].unsqueeze(0) #create a 2-Dim "fake" vector
        target = input.clone()
        if target.sum() > 0: # no user if he hasnt rated any movie at all
            output = sae(input)
            target.requires_grad_(False)
            output[target == 0] = 0 # no movies with "0" rating from a user
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.detach() > 0) + 1e-10)
            loss.backward() # With backward() you determine "in which direction" you have to go to minimize the error. Increase or decrease
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step() # With optimizer.step() you actually make the "step" in this direction -> update weights
    print("epoch: " +str(epoch)+" loss: " +str(train_loss/s))

# Testing the SAE 
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = training_set[id_user].unsqueeze(0)
    target = test_set[id_user].unsqueeze(0)
    if target.sum() > 0:
        output = sae(input)
        target.requires_grad_(False)
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.detach() > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print("test loss: " +str(test_loss/s))























