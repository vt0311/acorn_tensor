'''
Created on 2017. 12. 29.

@author: acorn
'''
# 채널 1개짜리 컨볼루션
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# input data 와 관련된 변수 목록
batch = 1 # 데이터 갯수
input_height = 3 # 높이
input_wight = 3 # 너비
input_channels = 1 # 채널 갯수(그림의 색상수)

data = range(input_height * input_width * input_channels)

image = np.reshape(data, ([batch, input_height, input_width, input_channels]))
image = image.astype(np .float32)

plt.imshow(image.reshape(3,3), cmap='Greys', interpolation='nearest')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# filter와 관련된 변수 목록
filter_height = 2
filter_width = 2
out_channels = 1

# filter 를 weight 라고 이해하면 됩니다.
# filter : HWCF( Height, Width, Channel, Filter)
# input_channels : 입력 데이터의 4번째 요소와 filter의 3번째 요소는 동일해야 한다.
w = tf.channels(([1.0, 1.0, 1.0, 1.0]))
w = tf.reshape(w, [filter_height, filter_width, input_channels, out_channels])







