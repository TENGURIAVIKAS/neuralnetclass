from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np 
import sys


leng=3
data = [[i+j for j in range(leng)] for i in range(100)]
target = [[i+j+1 for j in range(leng)] for i in range(1,101)]
data = np.array(data, dtype=np.float32)
target = np.array(target, dtype=np.float32)
data = data.reshape((100,1,leng))/200
target = target.reshape((100,1,leng))/200


x_test = [[i+j for j in range(leng)] for i in range(50,150)]
y_test = [[i+j+1 for j in range(leng)] for i in range(51,151)]
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
x_test = x_test.reshape((100,1,leng))/200
y_test = y_test.reshape((100,1,leng))/200



model = Sequential()
model.add(LSTM(leng, input_shape=(1,leng), return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1,leng), return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1,leng), return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1,leng), return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1,leng), return_sequences=True,activation='sigmoid'))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.fit(data,target,nb_epoch=2500,batch_size=50, verbose=2, validation_data=(x_test,y_test))

model.save("rnn_sequence.h5")
del model