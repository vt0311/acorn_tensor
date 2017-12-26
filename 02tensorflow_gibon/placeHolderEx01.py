'''
Created on 2017. 12. 22.

@author: acorn
'''
# placeHolderEx01.py
# y = 2 * x + 1 이라는 방정식을 placeholder을 이용하여 만들어 본 예시이다.
 
# placeholder를 이용한 간단한 1차 방정식
import tensorflow as tf
 
# 단계 1 : Build graph using TF operations
# x : 입력 데이터, y : 출력될 데이터
 
# placeholder란 치환 시켜야 할 미지의 어떤 값을 의미한다.
# 실행이 되는 시점에 치환이 되는 변수를 의미한다.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
 
x_data =  [ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 ]
y_data =  [ 3.1, 4.1, 4.9, 6.1, 6.9, 8.2, 9.1]
 
# 가중치와 바이어스의 초기 값을 설정한다.
w = tf.Variable(1.0)
b = tf.Variable(1.0)
 
# 가설을 만든다.
H = w *  x + b
 
# cost 함수를 작성한다.
diff = tf.square(H - y) # difference
cost = tf.reduce_mean(diff)
 
# Minimize : 경사 하강법에 의한 최소화 작업
learn_rate = 1e-3 #학습율
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)
 
# 단계 2,3 : Run/update graph and get results
 
# Launch the graph in a session.
sess = tf.Session() # 세션 객체를 구한다.
 
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
 
# feed_dict 인자 : placeholder에 대하여 입력해줘야 할 어떠한 정보를 넣어 주는 곳
# 파이썬의 사전 형식으로 입력을 해주면 된다.
for step in range(10000):
    _t, _w, _b, _c, _h = sess.run([train, w, b, cost, H], \
                                  feed_dict={x : x_data, y : y_data})
 
    print('step : %d, cost : %f, weight : %f, bias : %f' % \
        ( step, _c, _w, _b))
