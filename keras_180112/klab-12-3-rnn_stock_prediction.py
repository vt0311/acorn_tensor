# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import numpy as np
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# brew install graphviz

# pip3 install graphviz

# pip3 install pydot-ng
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt


# 7일(seq_length)간의 이전 데이터를 토대로 다음 날 예측하기
timesteps = seq_length = 7
data_dim = 5 # 입력 차원수(엑셀의 컬럼수)

# Open,High,Low,Close,Volume
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')

# 가장 최근 데이터가 위로 오게끔 함.
xy = xy[::-1]  # reverse order (chronically ordered)

# very important. It does not work without it.
# 정규화 작업
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)

x = xy # 엑셀 파일의 모든 열
print('x:', x)
y = xy[:, [-1]]  # Close as label  # 마지막 열

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length] # [0:7]?
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing
# (70 : 30) 나눔 
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

# 입력 데이터셋
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])

# 출력 데이터셋
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

model = Sequential()
model.add(LSTM(1, input_shape=(seq_length, data_dim), return_sequences=False))

# model.add(Dense(1))
model.add(Activation("linear")) # 값 예측, 즉 선형 회귀로 접근해야 한다.
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

# Store model graph in png
# (Error occurs on in python interactive shell)
#plot_model(model, to_file=os.path.basename(__file__) + '.png', show_shapes=True)

print(trainX.shape, trainY.shape)
model.fit(trainX, trainY, epochs=200)

# make predictions
testPredict = model.predict(testX)

# inverse values
# testPredict = scaler.transform(testPredict)
# testY = scaler.transform(testY)

# print(testPredict)
plt.plot(testY)  # 정답 그래프

plt.plot(testPredict)  # 예측한 그래프

plt.show()
