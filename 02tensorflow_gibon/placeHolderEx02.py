'''
Created on 2017. 12. 22.

@author: acorn
'''
# placeHolderEx02.py
# y = 3 * x + 2 이라는 방정식을 placeholder을 이용하여 만들어 본 예시이다.
 
# placeholder를 이용한 간단한 1차 방정식
import tensorflow as tf


# placeholder 란 치환시켜야 할 미지의 어떤 값을 의미한다.
# 실행이 되는 시점에 치환이 되는 변수를 의미한다.

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

x_data = [ 1, 1.5, 2, 2.5, 3]
y_data = [4.9, 6.6, 8.2, 9.3, 10.9]

w = tf.Variable(1.0)
b = tf.Variable(1.0)

H = w * x + b

diff = tf.square(H - y)
cost = tf.reduce_mean(diff)

learn_rate = 1e-3
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
train = optimizer.minimize( cost )
 
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
 
for step in range(10000):
    _t, _w, _b, _c, _h = sess.run( [train, w, b, cost, H], feed_dict={x:x_data, y:y_data} )
    print('step : %d, cost : %f, weight : %f, bias : %f' % \
        ( step, _c, _w, _b ))



