'''
Created on 2018. 1. 3.

@author: acorn
'''
# Activation Functions
#----------------------------------
#
# This function introduces activation
# functions in TensorFlow

# Implementing Activation Functions
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# X range : -10 부터 10까지 99등분하기
x_vals = np.linspace(start=-10., stop=10., num=100)

# ReLU activation : max(0, x)
print(sess.run(tf.nn.relu([-3., 3., 10.])))
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 activation :
# 매끄럽지 않은 약간 각이 진 시그모이드 함수와 유사한 모양
# min(max(0,x), 6)
print(sess.run(tf.nn.relu6([-3., 3., 10.])))
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# Sigmoid activation (시그모이드)
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent activation (하이퍼블릭 탄젠트)
print(sess.run(tf.nn.tanh([-1., 0., 1.])))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign activation : x/(abs(x) + 1)
print(sess.run(tf.nn.softsign([-1., 0., 1.])))
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus activation : 조금더 매끄러운 ReLU 함수
print(sess.run(tf.nn.softplus([-1., 0., 1.])))
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear activation
# 지수 선형 유닛 : Softplus 함수와 유사하고, 하부 점근선이 -1이다.
print(sess.run(tf.nn.elu([-1., 0., 1.])))
y_elu = sess.run(tf.nn.elu(x_vals))

# Plot the different functions
plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x_vals, y_elu, 'k-', label='ExpLU', linewidth=0.5)
plt.ylim([-1.5,7])
plt.legend(loc='upper left')
plt.show()

plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='upper left')
plt.show()