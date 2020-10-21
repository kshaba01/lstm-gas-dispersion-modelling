# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:50:48 2019

@author: k_sha
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import read_csv
import matplotlib.pyplot as plt
import os
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



#144 records, indexed 0 to 143
#our lookback is 1 meaning we need to stop at record 142 to be able to link it with the next record

#the function below is recasting the data as 
#X=t and Y=t+1
#where t = timestep
#NB this is timeseries data so we are trying to link X and X+1
#the -1 below is due to python 0 based indexing
#this function links X to X+1

#range(5) - last value is omitted...

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    #this will loop from 0 to 142
    for i in range(len(dataset)-look_back-1):
        #filter out the current record, append to X
        a = dataset[i:(i+look_back), 0] # row + features to append
        #print( 'X is' , i, i+look_back )
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
        #print('X is' , i, i+look_back , 'Y is' ,i+look_back)       
    return np.array(dataX), np.array(dataY) 


#fix random seed for reproduceability
np.random.seed(7)

#load the dataset
wd = (r'C:\Users\kehs\OneDrive - DNV GL\QRA Documents\QRAAI\Week 10 - Deep Learning 2 (RNN)-20191205')

os.chdir(wd)
dataframe = read_csv('8440401_image006_3.csv', usecols=[1])

dataset = dataframe.values

dataset = dataset.astype('float32')


#normalise the dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)


#split into train and test sets
train_size = int(len(dataset) * 0.67)
#print(train_size)
#96
test_size = len(dataset) - train_size
#print(test_size)
#48

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

#reshape the input into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#reshape input to be [samples, time steps, features]

trainX.shape


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#create and fit the LSTm network...


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=80, batch_size=1, verbose=0)

#make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


#invert predictions
#what does this mean? - Why are we doing this?

#this undoes the scaling applied to the input X i.e. we are undoing the scaling  between 0 to 1

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


#calculate RMSE

trainScore = (trainPredict - trainY)**2

print('Train Score: %.2f MSE' % (trainScore.mean())) 

testScore = (testPredict - testY)**2

print('Test Score: %.2f MSE' % (testScore.mean()))



# shift train predictions for plotting


#Return a new array with the same shape and type as a given array.
trainPredictPlot = np.empty_like(dataset) 


#Convert a string or number to a floating point number, if possible.
trainPredictPlot[:, :] = np.nan 

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# shift test predictions for plotting 

testPredictPlot = np.empty_like(dataset) 
testPredictPlot[:, :] = np.nan 
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict 

# plot baseline and predictions 
plt.plot(scaler.inverse_transform(dataset)) 
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot) 
plt.legend(["Base Data", "Train Data", "Test Data"])
#plt.show()



