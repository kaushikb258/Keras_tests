# neural net with keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import numpy as np
import sys


# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# normalize
for i in range(X.shape[1]):
 max1 = np.max(X[:,i])
 min1 = np.min(X[:,i])
 for j in range(X.shape[0]):
  X[j,i] = (X[j,i]-min1)/(max1-min1)
  if X[j,i] >1.0 or X[j,i] <0.0:
    print "error" , X[j,i]
    sys.exit()


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dropout(0.0))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.0))
model.add(Dense(1, activation='sigmoid'))

# Compile model
mymodel = 1

if mymodel == 1:
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif mymodel == 2:
 sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
 model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])



# Fit the model
model.fit(X, Y, nb_epoch=300, batch_size=20,verbose=2)

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x) for x in predictions]

fp = 0
tp = 0
fn = 0
tn = 0

for i in range(Y.shape[0]):
 if Y[i] ==0: 
  if rounded[i]==0:
   tn += 1
  else:
   fp += 1
 else:
  if rounded[i]==0:
   fn += 1
  else:
   tp += 1
  
print "------------------------------"
print "true positives: ", tp
print "true negatives:", tn
print "false positives: ", fp
print "false negatives: ", fn
print "accuracy: ", np.float(tp+tn)/np.float(tp+tn+fp+fn)

