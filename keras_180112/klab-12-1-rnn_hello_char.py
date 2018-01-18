import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation, LSTM
from keras.utils import np_utils

import os

# brew install graphviz
# pip3 install graphviz
# pip3 install pydot-ng
from keras.utils.vis_utils import plot_model

# sample text
sample = "hihello"

# unique한 글자들을 저장하고 있는 리스트
char_set = list(set(sample))  # id -> char ['i', 'l', 'e', 'o', 'h']

# 글자 1개와 인덱스 번호 1개를 가지고 있는 사전
char_dic = {w: i for i, w in enumerate(char_set)}

print('char_dic:', char_dic)

#   ----y_str--
# h i h e l l o 
# ---x_str---
x_str = sample[:-1] # hihell
y_str = sample[1:] # ihello

data_dim = len(char_set) # 입력 자원의 수
seq_length = len(y_str) # 시퀀스의 길이
num_classes = len(char_set) # 분류되는 클래스의 갯수

print(x_str, y_str)

# 각 글자들의 인덱스를 저장하고 있는 리스트
x = [char_dic[c] for c in x_str]  # char to index  (입력데이터)(x 0 1 0 2 4 4)
y = [char_dic[c] for c in y_str]  # char to index

# One-hot encoding 시킨다.
x = np_utils.to_categorical(x, num_classes=num_classes)

# samples는 batch size이다.
# reshape X to be [samples, time steps, features]
# len(x) : 시퀀스 길이, data_dim : 입력데이터의 차원(input dimension)
x = np.reshape(x, (-1, len(x), data_dim))
print(x.shape)

# One-hot encoding
y = np_utils.to_categorical(y, num_classes=num_classes)
# time steps
y = np.reshape(y, (-1, len(y), data_dim))
print(y.shape)

model = Sequential()

# LSTM : Long Short Term Memory (교재 417쪽)
# LSTM의 매개변수
# units : 양수, 출력의 차원
# input_shape : 입력할 shape(시퀀스길이, 입력의 차원)
# return_sequences : boolean, 마지막 출력 결과를 반환할 것인가
# activation : 활성화 함수
model.add(LSTM(num_classes, input_shape=(
    seq_length, data_dim), return_sequences=True))
model.add(TimeDistributed(Dense(num_classes)))
model.add(Activation('softmax'))
model.summary()
# Store model graph in png
# (Error occurs on in python interactive shell)
# plot_model(model, to_file=os.path.basename(__file__) + '.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
#model.fit(x, y, epochs=1)
model.fit(x, y, epochs=500)

# predict : 입력 데이터에 대한 출력 예측치를 배열 형식으로 반환해준다.
predictions = model.predict(x, verbose=0)

for i, prediction in enumerate(predictions):
    print(prediction)
    
    # x[i] : 각 입력 글자들의 인덱스를 지정한다.
    x_index = np.argmax(x[i], axis=1)
    
    # x_str : 각 색인에 해당하는 글자들의 모임
    x_str = [char_set[j] for j in x_index]
    print(x_index, ''.join(x_str))
    
    #[2 4 2 3 0 0] hihell
    #[4 0 3 0 0 1] ilello

    index = np.argmax(prediction, axis=1)
    result = [char_set[j] for j in index]
    print(index, ''.join(result))
