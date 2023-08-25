import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# Part 1 - Data Preprocessing
# Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values #gives values of the matrix in x row. It takes all the columns of the file außer die letzte
Y = dataset.iloc[:, -1].values # ":" sagt das alle Zeilen ausgegeben werden sollen und ", -1" das nur die letzte Spalte benutzt wird
# print(X)
# print(Y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

#One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(X_test)

# Feature Scaling / Normierung
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

# Part 2 - Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()  # Its from the models module from Karis Library (which belongs to TensorFlow 2.0)
# das stellt im Prinzp unser ANN dar als Sequentielle Layer Abfolge (Input,Hidden,Output)

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN
# Compiling the ANN
"""
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#if we do a binary classification and not a catergory classification
#(mean we predict a binary output (0 or 1)), the loss 'loss' function always has to be 'binary_crossentropy')
# if we would do a categorical or non-binary classification, we would have to use "loss = categorical_crossentropy".
# + take notice that than the output layer is not "sigmoid"-function anymore, but instead it would have to be "softmax"-funtion
"""

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model
"""
**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
**Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, 
"France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
