# https://github.com/fchollet/keras/tree/master/examples
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
from keras_180112.myfunction2 import getDataSet
from keras_180112.myfunction2 import getDataProp

# myfunction.printCategory() 함수 형태로 만들어 보기



# 엑셀 파일(data-04-zoo.csv)로 처리하기
# 비율로 처리하기 : 함수 getDataProp() 
#                 훈련용 데이터와 테스트용 데이터를 비율로 나눠주는 함수

# 함수 printCategory() :
# 결과 수치 데이터를 이용하여 종의 이름을 출력해주는 함수
# 종의 분류 : 강아지, 고양이, 치타, 코끼리, 사슴, 노루, 돼지



# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
#x_data = xy[:, 0:-1]
#y_data = xy[:, [-1]] - 1
#print(x_data.shape, y_data.shape)

nb_classes = 7

x_train, x_test, y_train, y_test = getDataProp(xy, testing_rate= 0.2, one_hot= True, num_classes= nb_classes)
y_one_hot = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()
model.add(Dense(nb_classes, input_shape=(16,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000)

#result = model.predict_classes(x, batch_size, verbose)

def printCategory(mydata):
    namelist = ['강아지', '고양이', '치타', '코끼리', '사슴', '노루', '돼지']
    for idx in mydata :
        print( namelist[idx], "(", idx, ")")
    

# Let's see if we can predict
pred = model.predict_classes(x_train)
for p, y in zip(pred, y_train):
    print("prediction: ", p, " true Y: ", y)
printCategory(pred)
