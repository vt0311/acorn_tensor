# Lab 7 Learning rate and Evaluation
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
np.random.seed(777)  # for reproducibility

#x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
#          [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]

x_data = np.array([[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
                   [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]])

#y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
#          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

y_data = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
                   [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

# 새로 추가 18.01.16
inputDim = x_data.shape[1]
myUnit = y_data.shape[1]

model = Sequential()
#model.add(Dense(3, input_dim=3))
model.add(Dense(units = myUnit, input_dim = inputDim))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=200)

predictions = model.predict(x_test)

print('==================')
print(predictions)
print('==================')

score = model.evaluate(x_test, y_test)

print('Prediction: ', [np.argmax(prediction) for prediction in predictions])

print('Cost:', score[0])

print('Accuracy: ', score[1])
