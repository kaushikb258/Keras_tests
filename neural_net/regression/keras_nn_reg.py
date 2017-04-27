# neural net with keras

import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
#from __future__ import print_function
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn import preprocessing


def read_data():
 X = pd.read_table("concrete.csv", sep=",",header=None)
 X = np.array(X)
 print X.shape
 rows = X.shape[0]
 cols = X.shape[1]
 X = normalize(X)

 x_train = []
 x_test = []
 y_train = []
 y_test = []

 for i in range(rows):
   if(random.random() < 0.9):
    x_train.append(X[i,:cols-1])
    y_train.append(X[i,cols-1])
   else: 
    x_test.append(X[i,:cols-1])
    y_test.append(X[i,cols-1])

 x_train = np.array(x_train)
 y_train = np.array(y_train)
 x_test = np.array(x_test)
 y_test = np.array(y_test)
 return x_train, y_train, x_test, y_test   


def normalize(X):
 for j in range(X.shape[1]):
  xmin = np.min(X[:,j])
  xmax = np.max(X[:,j])  
  for i in range(X.shape[0]):
   X[i,j] = (X[i,j]-xmin)/(xmax-xmin)  
 return X



if __name__ == '__main__':
 x_train, y_train, x_test, y_test = read_data()
 print "features = ", "cement, slag, ash, water, superplastic, coarseagg, fineagg, age"
 print "target = strength"

# neural net
model = Sequential()
model.add(Dense(input_dim=8, output_dim=7))
model.add(Activation('relu'))
model.add(Dense(input_dim=7, output_dim=6))
model.add(Activation('relu'))
model.add(Dense(input_dim=6, output_dim=5))
model.add(Activation('relu'))
model.add(Dense(input_dim=5, output_dim=1))
model.add(Activation('relu'))


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, nb_epoch=100, verbose=2, validation_data=(x_test, y_test), shuffle=True)

y_predict = model.predict(x_test, batch_size=1)


plt.plot(y_test,y_predict,'ro')
plt.xlabel('actual strength')
plt.ylabel('predicted strength')
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.savefig('concreteStrength.png',dpi=100)
plt.show()
