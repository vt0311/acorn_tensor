'''
Created on 2017. 12. 26.

@author: acorn
'''
# GradientDescent.py
import tensorflow as tf
import matplotlib.pyplot as plt
 
# y = w * x 인 직선의 방정식에 대하여 w의 값을 변화(-3.0 <= w < 5.0)시키면서
# 경사 하강법의 개념에 대하여 살펴 본다.
# w가 1.0이 사실 정답이다.
 
x = [1.0]
y = [1.0]
 
w = tf.placeholder(tf.float32)
 
H = x * w
 
cost = tf.reduce_mean(tf.square(H - y))
 
sess = tf.Session()
 
sess.run(tf.global_variables_initializer())
 
mylist = []
W_val = []
cost_val = []
 
for i in range(-30, 50):
    feed_W = i * 0.1 # (-3.0 <= w < 5.0)
    curr_cost, curr_W = sess.run([cost, w], feed_dict={w: feed_W})
    if i % 5 == 0 :
        sublist = []
        sublist.append(curr_W)
        sublist.append(curr_cost)
        mylist.append( sublist )
 
    W_val.append(curr_W)
    cost_val.append(curr_cost)
 
# Show the cost function
for item  in mylist : # weight, cost
    print ( item  )
 
# weidht와 cost를 그래프로 그린다.
plt.plot(W_val, cost_val)
plt.show()
