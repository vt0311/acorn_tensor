'''
Created on 2017. 12. 28.

@author: acorn
'''
NetworkXor02.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
 
# 이전에, 제대로 동작하지 않던 프로그램에 중간 layer을 하나 더 추가한 예시이다
# 3단 layer으로 만들어 보는 예시이다.
 
# neural network
# https://www.google.co.kr/search?q=neural+network&tbm=isch&imgil=DVoxuHT8e20ycM%253A%253BjZNjIe3vPrE87M%253Bhttp%25253A%25252F%25252Fcs231n.github.io%25252Fneural-networks-1%25252F&source=iu&pf=m&fir=DVoxuHT8e20ycM%253A%252CjZNjIe3vPrE87M%252C_&usg=__qqSvvg8f4XjqeyOv_0x_whtaTCA%3D&biw=1280&bih=918&ved=0ahUKEwiP6LPftI_UAhVDObwKHY0YB8AQyjcIcw&ei=GBgpWY_8IsPy8AWNsZyADA#imgrc=DVoxuHT8e20ycM:
 
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
 
# xor 연산이어서 2번 잘라야(Hidden layer를 두어야) 학습이 된다.
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
 
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
 
# weight를 2개 만든다고 가정하면
weight1_row = x_column # x의 열수와 동일해야 한다.
weight1_column = 2 # 개발자가 임의로 지정할 수 있다.
bias1 = weight1_column #w1의 열수와 동일해야 한다.
 
w1 = tf.Variable(tf.random_normal([weight1_row, weight1_column]) )
b1 = tf.Variable(tf.random_normal([bias1]) )
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
 
weight2_row = weight1_column # w1의 컬럼 수와 동일하다.
weight2_column = y_column # y의 열수와 동일해야 한다.
bias2 = weight2_column #w2의 열수와 동일해야 한다.
 
print('weight1_row :', weight1_row, ', weight1_column :', weight1_column)
print('weight2_row :', weight1_row, ', weight2_column :', weight2_column)
 
w2 = tf.Variable(tf.random_normal([weight2_row, weight2_column]) )
b2 = tf.Variable(tf.random_normal([bias2]) )
H = tf.sigmoid(tf.matmul(layer1, w2) + b2)
 
diff = y * tf.log(H) + ( 1 - y ) * tf.log( 1 - H )
cost = -tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)
 
# Accuracy computation
# True if H>0.5 else False
predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
 
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
 
   for step in range(10001):
       sess.run(train, feed_dict={x: x_data, y: y_data})
       if step % 100 == 0:
           _cost, _weight1, _weight2 = sess.run([cost, w1, w2], feed_dict={x: x_data, y: y_data})
           print('훈련 회수(step) :', step, '\n비용(cost) :', _cost, '\n가중치(weight1) :\n', _weight1, '\n가중치(weight2) :\n', _weight2)
           print('-----------------------------------------------------')
 
   # Accuracy report
   hypothesis, _predicted, _accuracy = sess.run([H, predicted, accuracy], feed_dict={x: x_data, y: y_data})
   print("\nHypothesis: \n", hypothesis, "\nCorrect: \n", _predicted, "\nAccuracy: ", _accuracy)