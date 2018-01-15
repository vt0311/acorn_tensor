'''
Created on 2018. 1. 15.

@author: acorn
'''
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from keras_180112.myfunction import getDataSet

import numpy as np

# 엑셀 파일로 다시 풀기
# 참조 파일 : myfunction.py 파일을 이용하기
# 엑셀 파일 :logistic02.csv

filename = './logistic02.csv'
data = np.loadtxt(filename, dtype=np.int32, delimiter=',')

x_train, x_test, y_train, y_test = getDataSet(data, testing_row= 2)

#x_data = [ [1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2] ]

#y_data = [ [0], [0], [0], [1], [1], [1] ]

model = Sequential()

# 출력(units) 1개, 입력(input_dim) 2개
# 활성화 함수는 시그모이드
model.add(Dense(units=1, input_dim = 2, activation='sigmoid'))

learning_rate = 0.1 # 학습율

# sgd = 옵티마이져 객체
sgd = SGD(lr= learning_rate)

# binary_crossentropy는 이항 분류 시 사용되는 손실함수 
model.compile(optimizer=sgd, loss='binary_crossentropy')

model.fit(x = x_train , y = y_train, epochs= 2000)

print(model.predict_classes(x_test))
    
#for item in x_test :
#    print(model.predict_classes(np.array[item]))
    #print(item)

#print(model.predict_classes(np.array([[2,1]])))

#print(model.predict_classes(np.array([[6,5]])))

