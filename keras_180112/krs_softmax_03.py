# krs_softmax_03.py

# 엑셀 파일로 처리(data-04-zoo.csv)하기
# 함수 getDataProp()
#       훈련용 데이터와 테스트용 데이터를 비율로 나눠 주는 함수  

# 함수 printCategory() :
# 결과 수치 데이터를 이용하여 종의 이름을 출력해주는 함수
# 종의 분류 : 강아지,고양이,치타,코끼리,사슴,노루,돼지

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

from keras_180112.myfunction3 import getDataProp, printCategory

nb_classes = 7

filename = './data-04-zoo.csv'

data = np.loadtxt(filename, dtype=np.int32, delimiter=',')

x_train, x_test, y_train, y_test = \
    getDataProp(data, testing_rate = 0.2, \
            one_hot= True, \
            num_classes = nb_classes )

model = Sequential()
model.add(Dense(units = nb_classes, input_shape=(16,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000)
    
namelist = ['강아지', '고양이', '치타', '코끼리', '사슴', '노루', '돼지']
result = model.predict_classes(x_test)
printCategory(namelist, result )