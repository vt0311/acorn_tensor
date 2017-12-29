'''
Created on 2017. 12. 29.

@author: acorn
'''
# 채널 1개짜리 컨볼루션
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

batch = 1 # 데이터 갯수

data = range(input_height * input_width * input_channels)

image = np.reshape(data, ([batch, input_height, input_width, input_channels]))
image = image.astype(np.float32)

plt.imshow(image.reshape(3,3), cmap='Greys', interpolation='nearest')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

