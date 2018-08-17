

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#used to create dummy variable as country had 3 types 0,1,2 to give them equal weightage
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#removing the first column because of dummy variable for 3 only 2 are required
X=X[:,1:]

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#ann
import keras
#used to initialize ann
from keras.models import Sequential
# used to built layers
from keras.layers import Dense

#classifier class is used
classifier=Sequential()

#adding the input and hidden layer
#function used
#rectifier -hidden && sigmoid - output
#output value=6 avg of X and Y (11+1)/2=6
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#output layer-
#binary outcome only 1 output
#for more than 1 output softmax(sigmoid funcion which is for more than variable) function is used 
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ann
#optimize-to find optimal weight algorithm used is adam (type of scothsatic gradient)
#loss- to calculate cost function
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ann-making connection between input and ann
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)
