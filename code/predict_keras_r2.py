from keras.models import load_model
from keras.models import Sequential
import numpy as np 

leng=3
x_test = [[i+j for j in range(leng)] for i in range(50,150)]
y_test = [[i+j+1 for j in range(leng)] for i in range(51,151)]
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
x_test = x_test.reshape((100,1,leng))/200
y_test = y_test.reshape((100,1,leng))/200

model = load_model("rnn_sequence.h5")

predict = model.predict(x_test)
print(predict[0,0,:]*200)
print(y_test[0,0,:]*200)