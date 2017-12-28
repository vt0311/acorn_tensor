'''
Created on 2017. 12. 27.

@author: acorn
'''
# softMaxPreTest02.py
# tensorflow의 argmax 함수에 대한 사용 예시이다.
# argmax() 함수는 해당 요소들 중에서 최대 값에 해당하는 항목의 인덱스를 반환해준다.
import tensorflow as tf
import numpy as np
 
mylist = [[1, 2, 3], [8, 6, 2]]
yourlist = [[1, 2, 4], [5, 8, 2]]
 
# axis = 1 : 행에서 최대 값을 갖는 인덱스 번호
row1 = tf.argmax(mylist, axis = 1)
row2 = tf.argmax(yourlist, axis = 1)
 
# axis = 0 : 열에서 최대 값을 갖는 인덱스 번호
column = tf.argmax(mylist, axis = 0 )
 
sess = tf.Session()
# [2 0] : 0번째 행에서 가장 큰 수는 2번째에 있다.
#         1번째 행에서 가장 큰 수는 0번째에 있다.
 
print('행에서 최대 값을 갖는 인덱스 번호 찾기')
print(sess.run(row1))
print('-------------------------------------------------')
print('열에서 최대 값을 갖는 인덱스 번호 찾기')
print(sess.run(column)) # [1 1 0]
print('-------------------------------------------------')
test = [np.exp(2.0), np.exp(1.0), np.exp(0.1)]
result = tf.argmax(test, axis = 0 )
print('test 변수에서 가장 큰 수가 있는 인덱스는 %d번이다.' %  sess.run(result)) # 0
print('-------------------------------------------------')
print('tf.equal 메소드 사용하기')
print(sess.run(tf.equal(row1, row2)))  