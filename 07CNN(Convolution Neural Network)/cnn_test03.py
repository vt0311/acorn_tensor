'''
Created on 2017. 12. 29.

@author: acorn
'''
# 채널 2개짜리 컨볼루션
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# input data 와 관련된 변수 목록
batch = 1 # 데이터 갯수(이미지의 갯수)
input_height = 3 # 높이
input_width = 3 # 너비
input_channels = 2 # 채널 갯수(그림의 색상수)

data = range(input_height * input_width * input_channels)

image = np.reshape(data, ([batch, input_height, input_width, input_channels]))
image = image.astype(np.float32)

#plt.imshow(image.reshape(3,3), cmap='Greys', interpolation='nearest')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# filter와 관련된 변수 목록
filter_height = 2
filter_width = 2
out_channels = 4

# input : NHWC(1332)
# filter 를 weight 라고 이해하면 됩니다.
w = tf.constant(([
    [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0],
    [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]
    ]))
# filter 를 weight 라고 이해하면 됩니다.
# filter : HWCF( Height, Width, Channel, Filter)
w = tf.reshape(w, [filter_height, filter_width, input_channels, out_channels])

# stride와 관련된 변수
stride_width = 1
stride_height = 1

conv2d = tf.nn.conv2d(image, w, strides=[1, stride_width, stride_height, 1], padding='SAME')
conv2d_img = sess.run((conv2d))

print('\nimage :', image)

print('\nconv2d_img :', conv2d_img)

print('\nimage.shape :', image.shape)

print('\nconv2d_img.shape :', conv2d_img.shape)





