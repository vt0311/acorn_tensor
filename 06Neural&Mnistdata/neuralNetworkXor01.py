'''
Created on 2017. 12. 28.

@author: acorn
'''
# neuralNetworkXor01.py
 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
 
# xor 연산은 제대로 동작하지 않는다.
 
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32) # 4행 2열
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) # 4행 1열
 
# 중첩 리스트의 열 갯수 구하는 함수
def col_length( input ):
    sublist = input[0]
    length = len(sublist)
    return length
 
x_column = col_length( x_data )
print( '컬럼수 : ', x_column )
y_column = col_length( y_data )
print( '클래스 개수 : ', y_column )
 
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
 
weight_row = x_column # x의 열수와 동일해야 한다.
weight_column = y_column # y의 열수와 동일해야 한다.
bias = weight_column #w의 열수와 동일해야 한다.
 
w = tf.Variable(tf.random_normal([weight_row, weight_column]) )
b = tf.Variable(tf.random_normal([bias]) )
 
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(x, w)))
H = tf.sigmoid(tf.matmul(x, w) + b) # 4행 1열
 
diff = y * tf.log(H) + ( 1 - y ) * tf.log( 1 - H )
cost = -tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)
 
# Accuracy computation, True if H>0.5 else False
predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
 
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
 
   for step in range(10001):
       sess.run(train, feed_dict={x: x_data, y: y_data})
       if step % 100 == 0:
           _cost, _weight = sess.run([cost, w], feed_dict={x: x_data, y: y_data})
           print('훈련 회수(step) :', step, '\n비용(cost) :', _cost, '\n가중치(weight) :\n', _weight)
           print('-----------------------------------------------------')
 
   # Accuracy report
   hypothesis, _predicted, _accuracy = sess.run([H, predicted, accuracy], feed_dict={x: x_data, y: y_data})
   print("\nHypothesis: \n", hypothesis, "\nCorrect: \n", _predicted, "\nAccuracy: ", _accuracy)