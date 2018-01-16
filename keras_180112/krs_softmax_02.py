'''
Created on 2018. 1. 15.

@author: acorn
'''

# klab-06-1-softmax.py 파일을 변형하여 엑셀 파일로 처리하기.
# 엑셀 파일 : softmax02.csv
# np_utils.to_categorical( y데이터, nb_classes )

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
from keras_180112.myfunction import getDataSet

filename = './softmax02.csv'
data = np.loadtxt(filename, dtype=np.int32, delimiter=',')

nb_classes = 3
x_train, x_test, y_train, y_test = getDataSet(data, testing_row= 2, one_hot= True, num_classes= nb_classes)

#y = np.array([[2], [0], [1]], dtype=np.float32)

y_one_hot = np_utils.to_categorical(y_train, nb_classes)
print(y_one_hot)

model = Sequential()
model.add(Dense(nb_classes, input_shape=(4,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000)

def getClassName(mydata):
    namelist = ['강아지', '고양이', '토끼']
#    for item in mydata :
#        print(namelist[item])
    #for idx in range(len(mydata)) :
    for idx in mydata :
        print( namelist[idx], "(", idx, ")")
    #print(mydata)
    
result = model.predict_classes(x_test)
getClassName(result)  

'''
pred = model.predict_classes(x_train)
for p, y in zip(pred, y_train):
    print("prediction: ", p, " true Y: ", y)
getClassName(pred)    
'''

#print('history 객체:') #history 작업 모든기록이 들어있는 keras내부의 history
#print('history.on_epoch_begin(200):', history.on_epoch_begin(200))

'''
pred = model.predict_classes(x_train)
for p, y in zip(pred, y_one_hot):
    print("prediction: ", p, " true Y: ", y)
'''    
#print(model.predict_classes(np.array([[1, 2, 1, 1]])))
#print(model.predict_classes(np.array([[1, 2, 5, 6]])))