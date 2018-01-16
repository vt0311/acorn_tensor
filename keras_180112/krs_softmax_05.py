'''
Created on 2018. 1. 15.

@author: acorn
'''

# krs_softmax_05.py

# 엑셀 파일로 처리(weather_train.csv)하기
# 함수 getDataProp()
#       훈련용 데이터와 테스트용 데이터를 비율로 나눠 주는 함수  

# 함수 printCategory() :
# 결과 수치 데이터를 이용하여 종의 이름을 출력해주는 함수
# 종의 분류 : 

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


nb_classes = 1

filename = './weather_train.csv'
data1 = np.loadtxt(filename, dtype=np.int32, delimiter=',', skiprows=1)

filename2 = './weather_test.csv'
data2 = np.loadtxt(filename2, dtype=np.int32, delimiter=',', skiprows=1)

x_train = data1[:,2]
x_test = data2[:,2] 
y_train = data1[:,10]
y_test = data2[:,10]
            #num_classes = nb_classes 
  
print('x_train:', x_train)  
print('x_test:',x_test)  
print('y_train:',y_train)  
print('y_test:',y_test)  
  
  
# inputDim = x_train.shape[1]
# print(inputDim)
#   
model = Sequential()
model.add(Dense(units = nb_classes, input_shape=(1,)))
model.add(Activation('softmax'))
  
model.summary()
  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  
history = model.fit(x_train, y_train, epochs=1000)
      
namelist = ['Not Rain','Rain']
result = model.predict_classes(x_test)

def printCategory(result):
    for idx in result :
        print( result[idx], '(', idx, ')' )  
  
#printCategory(namelist)



print('--------------')
for prediction, answer in zip(result, x_test) :
     print('예측치:', prediction, ', 정답:', answer)