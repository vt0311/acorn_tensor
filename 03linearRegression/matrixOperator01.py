'''
Created on 2017. 12. 26.

@author: acorn
'''

# matrixOperator01.py
import tensorflow as tf
 
# 입력되는 항목이 1개가 아닌 여러 개인 경우의 처리는 행렬을 사용하여 가설을 세운다.
# Multi-variable linear regression(Hypothesis using matrix)
# x_data : shape(5, 3)
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
 
# y_data : shape(5, 1)
y_data = [[152.], [185.], [180.], [196.], [142.]]
 
# x는 몇행인지는 모르지만, 3열이다.
x_column = 3 # 입력 데이터의 컬럼 갯수
y_column = 1
x = tf.placeholder( tf.float32, shape=[None, x_column]) # ?행 3열
y = tf.placeholder( tf.float32, shape=[None, y_column]) # ?행 1열
 
# w는 가중치가 3개이므로 x_column행 y_column열이어야 한다.
w = tf.Variable( tf.random_normal([x_column, y_column])) # 3행 1열
b = tf.Variable( tf.random_normal([1]))
 
# H = (5행 3열)*(# 3행 1열)==>(# 5행 1열)
H = tf.matmul(x, w) + b # matmul(매트릭스 연산)
 
diff = tf.square( H - y )
cost = tf.reduce_mean( diff )
 
learn_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer( learning_rate = learn_rate )
train = optimizer.minimize( cost ) #학습
 
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
 
for step in range(20001) :
    _c, _h, _t = sess.run([cost, H, train], feed_dict={x : x_data, y : y_data})
    if step % 500 == 0 :
        print("step: ", step, "\nCost: ", _c, "\nPrediction:\n", _h)
        print('----------------------------')