'''
Created on 2018. 1. 3.

@author: acorn
'''

# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Using Placeholders
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x) # 입력 데이터(x)와 동일한 형식을 shape로 만들어주는 함수

print('x.shape', x.shape)
print('y.shape', y.shape)
# 임의의 값으로 채워진 4행 4열의 배열 만들기
rand_array = np.random.rand(4, 4)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph)

print(sess.run(y, feed_dict={x: rand_array}))
