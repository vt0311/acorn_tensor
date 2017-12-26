'''
Created on 2017. 12. 22.

@author: acorn
'''
# RankShapesType.py
# Tensor의 Ranks, Shapes, and Types 등의 정보를 확인하기
import tensorflow as tf
 
sess = tf.Session()
 
a = tf.constant(1.0) # 스칼라, 단순한 숫자 1개
print('a = ', a, '/ 값 : ', sess.run(a))
 
b = tf.constant(2.0, dtype=tf.float32)
print('b = ', b, '/ 값 : ', sess.run(b))
 
c = a + b
print('c = ', c, '/ 값 : ', sess.run(c))
 
d = tf.add(a, b)
print('d = ', d, '/ 값 : ', sess.run(d))
 
e = tf.constant([1,2,3]) # rank : 1, shape(3)
print('e = ', e, '/ 값 : ', sess.run(e))
 
f = tf.constant([[1,2,3],[4,5,6]]) # rank : 2, shape(2, 3)
print('f = ', f, '/ 값 : ', sess.run(f))
 
# g : rank : 3, shape(2, 1, 3)
g = tf.constant([[[1,2,3]],[[4,5,6]]])
print('g = ', g, '/ 값 : ', sess.run(g))
 
# h : rank : 3, shape(2, 2, 3)
h = tf.constant([[[0,0,0],[1,2,3]],[[1,1,1],[4,5,6,]]])
print('h = ', h, '/ 값 : ', sess.run(h))
