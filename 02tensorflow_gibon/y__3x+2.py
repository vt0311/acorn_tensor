# y__3x+2.py
# 간단한 1차 방정식
import tensorflow as tf
import numpy as np
 
# 최적의 w와 b를 기계 학습 시켜보세요.

# 비용 함수와 가중치의 변화율을 그래프로 표현하세요. 
 
# 단계 1 : Build graph using TF operations
# x : 입력 데이터, y : 출력될 데이터
# x = [1.0, 2.0, 3.0]
x = np.arange( 1.0, 3.1, 0.5 )
y = [4.9, 6.6, 8.2, 9.3, 10.9]
# 정답 y = [5, 6.5, 8, 9.5, 11]
 
print( 'x = ', x )
print( 'y = ', y )
 
# 가중치와 바이어스의 초기 값을 설정한다.
w = tf.Variable(0.1)
b = tf.Variable(0.1)
 
# 가설을 만든다.
H = w * x + b
 
# cost 함수를 작성한다.
diff = tf.square( H - y )
cost = tf.reduce_mean( diff )
 
# Minimize : 경사 하강법에 의한 최소화 작업
learn_rate = 1e-3
optimizer = tf.train.GradientDescentOptimizer( learning_rate = learn_rate )
train = optimizer.minimize( cost )
 
# 단계 2,3 : Run/update graph and get results
# Launch the graph in a session.
sess = tf.Session()
 
# Initializes global variables in the graph.
sess.run( tf.global_variables_initializer() )
 
for step in range(10000):
    sess.run( train )
    print('step : %d, cost : %f, w : %f, b : %f' % \
        ( step, sess.run(cost), sess.run(w), sess.run(b)))
    
    # w:2.9821
    # b:2.0071
import matplotlib.pyplot as plt    
 
cost_list = []
weight_list = []
for step in range(10000):
    sess.run( train )
    cost_list.append(sess.run(cost))
    weight_list.append(sess.run(w))
#return cost_list    
    
    
plt.plot(cost_list, 'b')
plt.show()
plt.plot(weight_list, 'r')
plt.show()