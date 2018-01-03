'''
Created on 2018. 1. 3.

# 계산 그래프의 연산(쿡북 64)
# placeholder를 사용한 계산 그래프의 연산
# 구구단 3단
@author: acorn
'''
# Operations on a Computational Graph
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
x1 = np.array(np.linspace(start=1., stop=9., num=9 ))
print(x1)

x_vals = np.array([1., 3., 5., 7., 9.]) # 입력할 데이터 셋
x_data = tf.placeholder(tf.float32) # 입력을 위한 플레이스 홀더
m = tf.constant(3.) # 단수 : 3단

# Multiplication
prod = tf.multiply(x_data, m) # 어떤 수 * 3
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))

merged = tf.summary.merge_all()
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)