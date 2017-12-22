# y_2x+1.py
# 간단한 1차 방정식
import tensorflow as tf
import numpy as np
 
# 단계 1 : 모델을 작성한다.
# 1.1 : 입력될 데이터와 출력될 데이터를 설정한다.
# x : 입력 데이터, y : 출력될 데이터
# 1.0부터 4.5까지 0.5씩 증가시키기
x = np.arange( 1.0, 4.1, 0.5 )
print( 'x = ', x ) # x =  [ 1.   1.5  2.   2.5  3.   3.5  4. ]
 
y = [ 3.1,  4.1,  4.9,  6.1,  6.9,  8.2,  9.1]
print( 'y = ', y ) # y =  [3.1, 4.1, 4.9, 6.1, 6.9, 8.2, 9.1]
 
# 1.2 가중치와 바이어스의 초기 값을 설정한다.
w = tf.Variable(0.1)
b = tf.Variable(0.1)
 
# 1.3 가설을 만든다.
H = w *  x + b
 
# 1.4 cost 함수를 작성한다.
diff = tf.square(H - y)#오류
cost = tf.reduce_mean(diff) #모든 샘플의 오류 평균
 
# 1.5 경사 하강법에 의한 최소화 작업을 수행한다.
learn_rate = 1e-3 #학습율
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)  #학습
 
# 단계 2,3 : Run/update graph and get results
 
# 2.1 세션 객체를 만든다.
sess = tf.Session() # 세션 객체를 구한다.
 
# 2.2 그래프 내의 글로별 변수들을 초기화한다.
sess.run(tf.global_variables_initializer())
 
# 2.3 그래프 내의 변수들을 업데이트 하면서 실행한다.
for step in range(10000):
    sess.run(train)
    print('step : %d, cost : %.12f, weight : %f, bias : %f' % \
        ( step, sess.run(cost), sess.run(w), sess.run(b)))
    
    
import matplotlib.pyplot as plt

cost_list = []
weight_list = []

#plt.plot(cost, 'b')    

    
    
    
    
    
    