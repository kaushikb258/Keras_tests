# LSTM for Apple stock price

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import datetime as dt
import pandas_datareader.data as web
import warnings
import sys

warnings.simplefilter("ignore")

start = dt.datetime(2014,7,1)
end = dt.datetime(2017,4,26)

# AAPL
aapl = web.DataReader("AAPL", 'yahoo', start, end)
aapl = np.array(aapl)
print aapl.shape
# columns
#Open High Low Close Volume Adj-Close



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back)]
    dataX.append(a)
    dataY.append(dataset[i + look_back])
  return np.array(dataX), np.array(dataY)



# normalize the dataset
col = 3
min1 = np.min(aapl[:,col])
max1 = np.max(aapl[:,col])
data = np.zeros((aapl.shape[0]),dtype=np.float32)
for i in range(aapl.shape[0]): 
 data[i] = (aapl[i,col]-min1)/(max1-min1) 


# split into train and test sets
train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train = data[0:train_size]
test = data[train_size:len(data)]
print "train size: ", train.shape
print "test size: ", test.shape


# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX1 = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX1 = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


print trainX.shape
print trainY.shape


nepochs = 25

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX1, trainY, epochs=nepochs, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX1)
testPredict = model.predict(testX1)

trainPredict = np.reshape(trainPredict,(trainPredict.shape[0]))
testPredict = np.reshape(testPredict,(testPredict.shape[0]))


trainPredict = (max1-min1)*trainPredict + min1
testPredict = (max1-min1)*testPredict + min1
trainX = (max1-min1)*trainX + min1
testX = (max1-min1)*testX + min1


x1 = np.zeros(trainX.shape[0])
x2 = np.zeros(testX.shape[0]) 
for i in range(trainX.shape[0]):
 x1[i] = i+1
for i in range(testX.shape[0]):
 x2[i] = trainX.shape[0] + i



plt.plot(x1,trainPredict,'g-',label='predicted (train)')
plt.plot(x1,trainX[:,0],'r-', label='train')
plt.plot(x2,testPredict,'y-',label='predicted (test)')
plt.plot(x2,testX[:,0],'b-',label='test')
plt.xlabel('trading day (start from 7/1/2014)',fontsize=15)
plt.ylabel('AAPL stock price',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
#plt.show()
plt.savefig('aapl.png',dpi=100)
