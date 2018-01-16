from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler
import numpy as np
np.random.seed(777)  # for reproducibility


xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# very important. It does not work without it.
# MinMaxScalar 클래스 : 0 과 1사이의 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)
print(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

inputDim = x_data.shape[1]
myUnit = y_data.shape[1]

model = Sequential()

#model.add(Dense(1, input_dim=4))
model.add(Dense(units = myUnit, input_dim = inputDim))
model.add(Activation('linear'))

model.summary()

model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=100)

predictions = model.predict(x_data)

# evaluate : 입력 데이터에 대하여 일괄 처리 단위로 손실 함수 결과와 정확도를 반환해준다. 
score = model.evaluate(x_data, y_data)


# 현대 이 모델 변수(model)가 가지고 있는 지표들을 문자열 형태로 보여준다.
print(model.metrics_names)
# ['loss', 'acc']


print('Prediction: \n', predictions)
print('Cost: ', score[0])
print('Accuracy: ', score[1])