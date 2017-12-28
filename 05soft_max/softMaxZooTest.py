'''
Created on 2017. 12. 28.

@author: acorn
'''
# softMaxZooTest.py

# 16개의 입력 정보를 이용하여 7개의 클래스 중 1개로 분류해주는 예시
import tensorflow as tf
import numpy as np
 
data = np.loadtxt('data-04-zoo-mini.csv', delimiter=',', dtype=np.float32)
# data = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
 
# print (data.shape)
table_row = data.shape[0]  # 엑셀 파일에서 읽어 들인 데이터의 총 행수
table_col = data.shape[1]  # 엑셀 파일에서 읽어 들인 데이터의 총 열수
column = table_col - 1  # 입력 데이터의 컬럼 갯수
 
testing_row = 3  # 테스트 용 데이터 셋의 행수
training_row = table_row - testing_row  # 훈련용 데이터 셋의 행수
 
nb_classes = 7  # 클래스의 갯수
 
print('입력 데이터의 컬럼 갯수(column) : %d' % (column))
print('클래스의 갯수(nb_classes) : %d' % (nb_classes))
print('table_row : %d, training_row : %d' % (table_row, training_row))
 
x_train = data[0:training_row, 0:column]
y_train = data[0:training_row, column:(column + 1)]
 
x_test = data[training_row:, 0:column]
y_test = data[training_row:, column:(column + 1)]
 
print('-------------------------------------------------------------------------')
print('훈련용 입력 데이터(x_train):\n', x_train)
print('-------------------------------------------------------------------------')
print('훈련용 정답 데이터(y_train):\n', y_train)
print('-------------------------------------------------------------------------')
print('테스트용 입력 데이터(x_test):\n', x_test)
print('-------------------------------------------------------------------------')
print('테스트용 정답 데이터(y_test):\n', y_test)
 
x = tf.placeholder(tf.float32, [None, column])
y = tf.placeholder(tf.int32, [None, 1])
 
w = tf.Variable(tf.random_normal([column, nb_classes]) )
b = tf.Variable(tf.random_normal([nb_classes]) )
 
Y_one_hot = tf.one_hot(y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
 
logits = tf.matmul(x, w) + b
H = tf.nn.softmax( logits )
 
diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)
 
# 컴퓨터가 예측한 값(prediction)
prediction = tf.argmax(H, axis = 1)
 
print('-------------------------------------------------------------------------')
 
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
 
   for step in range(2000): #2000
       sess.run(train, feed_dict={x: x_train, y: y_train})
 
       if step % 100 == 0:
           _yonehot, _c, _h, _p  = \
               sess.run([Y_one_hot, cost, H, prediction], \
                    feed_dict={x: x_train, y: y_train})
           if step == 0 :
                print("one hot 결과\n", _yonehot)  #
                print('-------------------------------------------------------------------------')
 
           print("훈련 회수: {:5}\t비용 : {:.3f}".format(step, _c))
           print("가설\n", _h) #
           print("컴퓨터가 예측한 값(prediction)", _p)  #
           print('-------------------------------------------------------------------------')
 
 
   pred = sess.run(prediction, feed_dict={x: x_train})
   for p, y in zip(pred, y_train.flatten()):
       print("[{}] Prediction: {}, Answer y: {}".format(p == int(y), p, int(y)))
 
   _h2 = sess.run([H], feed_dict={x: x_test})
   print( np.argmax(_h2, axis=1))
