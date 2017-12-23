import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")



filename = '130N_Cycles_1-47.xlsx'
# filename = 'sinwave.csv'
filetype = 'xlsx'
if filetype != 'csv':
	print("Reading xls file")
	data_xls = data_xls = pd.read_excel('130N_Cycles_1-47.xlsx',sheetname='Specimen_RawData_1',skiprows=[0])
	data_xls.drop("(sec)",inplace=True,axis=1)
	print("Converting xls to csv")
	data_xls.to_csv('forceCycles.csv', encoding='utf-8',index = False, header=False)

num_features = 300
weights_filepath = './best_weights_'+ str(num_features)  + '.hdf5'
checkpointer = ModelCheckpoint(filepath=weights_filepath,save_best_only=True)

filename = 'forceCycles.csv'
#seq_len = 50
seq_len = num_features

print("Reading csv file")
f = open(filename, 'rb').read()
data = f.decode().split('\n')

sequence_length = seq_len+1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])


result = np.array(result).astype(float)
print("The shape of result is {}".format(result.shape))

#max_el = np.max(result)
#print("maximum element is {}".format(max_el))

#min_el = np.min(result)
#print("minimum element is {}".format(min_el))

#diff = max_el - min_el
#print("diff is {}".format(diff))

#result = result.astype(float)/(diff)
result = result.astype(float)

X_train = result[:,:-1]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

y_train = result[:,-1]

def build_model(layers):
    model = Sequential()
    model.add(LSTM(
        input_shape = (layers[1], layers[0]),
        #output_dim = layers[1],
	output_dim = layers[2],

        return_sequences= True))
    model.add(Dropout(0.2))
    
    #model.add(LSTM(
    #    layers[2],
    #    return_sequences=False))
    #model.add(Dropout(0.2))
    
    model.add(Dense(
        output_dim = layers[3]))
    model.add(Activation("linear"))
    
    start = time.time()
    model.compile(loss = "mse", optimizer = "adam")
    print("Compilation time: ", time.time() - start)
    return model

epochs = 20

#layers = [1,50,100,1]
layers = [1,num_features,100,1]

print("Building model")
model = build_model(layers)
# print("model's weights before")
# print(model.get_weights())

print("Training the model")
model.fit(X_train, y_train, batch_size = 1024, nb_epoch = epochs, validation_split = 0.05,callbacks = [checkpointer])

# print("model's weights after")
# print(model.get_weights())

# print(X_train[-1])
# print(X_train.shape)

# print("next prediction")
# print(X_train[-1].shape)
# print(model.predict(X_train[-1]))
# print(model.input_shape)
X_test = X_train[-1].reshape(1,X_train[-1].shape[0],X_train[-1].shape[1])
# print("X_test shape = {}".format(X_test.shape))
l = model.predict(X_test)
# print(l)
# print("X_train shape = {}".format(X_train.shape))
# print("X_train-1 shape = {}".format(X_train[-1].shape))

data_points = X_train[-1]
data_points = data_points.reshape(1,data_points.shape[0],data_points.shape[1])
# print("data points shape: {}".format(data_points.shape))
prediction = model.predict(data_points)

print("Learning the predictions")
prediction_points = []
for i in range(1000):

	prediction = model.predict(data_points)
	# print(prediction.shape)
	# print(data_points.shape)
	# print(data_points)
	data_points = data_points[0,1:,0]
	# print("yay1 {}".format(data_points.shape))
	data_points = list(data_points)
	# print("yay2 {}".format(data_points))
	# print("yay3 {}".format(prediction))
	data_points.append(prediction[0][0])
	# print("yay4 {}".format(data_points))
	# data_points = data_points.append(prediction[0][0])
	data_points = np.array(data_points)
	# print(data_points)
	# print(data_points.shape)
	data_points = data_points.reshape(1,data_points.shape[0],1)
	#prediction_points.append(diff*prediction[0][0])
	prediction_points.append(prediction[0][0])
	# print("yay5 {}".format(data_points))
	# print("yay6 {}".format(data_points.shape))

#end for loop


print("Predictions are")
print(prediction_points)
# prediction_points = []
# for i in range(10):
# 	print("yay")
# 	prediction = model.predict(data_points)
# 	data_points = list(data_points)
# 	data_points.append(prediction[0][0])
# 	data_points = data_points[1:]
# 	data_points = np.array(data_points)
# 	print(data_points)
# 	data_points = data_points.reshape(1,data_points.shape[0],1)
# print(prediction_points)
#plt.plot(prediction_points)
#plt.show()
