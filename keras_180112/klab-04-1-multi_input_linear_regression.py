from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, adam
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

model = Sequential()
model.add(Dense(input_dim=3, units=1)) # input_dim : 입력의 차원
model.add(Activation('linear'))

rmsprop = adam(lr=1e-2)  
#rmsprop = RMSprop(lr=1e-10)  # RMS도 옵티마이져다.
model.compile(loss='mse', optimizer=rmsprop)
#model.fit(x_data, y_data, epochs=1000)
#model.fit(x_data, y_data, epochs=5000)
model.fit(x_data, y_data, epochs=10000)

# [95, 100, 80]을 예측해보아라.
y_predict = model.predict(np.array([[95., 100., 80]]))
print(y_predict)


