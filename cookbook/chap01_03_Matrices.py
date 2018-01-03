'''
Created on 2018. 1. 3.

@author: acorn
'''
# Matrices and Matrix Operations (쿡북 p.37)
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Declaring matrices
sess = tf.Session()

# Declaring matrices

# Identity matrix
# 3행 3열의 단위 행렬 생성하기
identity_matrix = tf.diag([1.0,1.0,1.0])
print(sess.run(identity_matrix))

# 2x3 random norm matrix
A = tf.truncated_normal([2,3])
print(sess.run(A))

# 2x3 constant matrix
# 2행 3열을 모두 5로 채우기
B = tf.fill([2,3], 5.0)
print(sess.run(B))

# 3x2 random uniform matrix
# 균등 분포를 따르는 난수 값으로 텐서를 생성한다.
C = tf.random_uniform([3,2])
print(sess.run(C))
print(sess.run(C)) # Note that we are reinitializing, hence the new random variabels

# Create matrix from np array
# convert_to_tensor : 배열을 텐서 객체로 만들어 주는 함수
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# Matrix addition/subtraction
print(sess.run(A+B))
print(sess.run(B-B))

# Matrix Multiplication
print(sess.run(tf.matmul(B, identity_matrix))) # 행렬 연산

# Matrix Transpose 행렬 전치
print(sess.run(tf.transpose(C))) # Again, new random variables

# Matrix Determinant 행렬식 구하기
print('\n Matrix Determinant D:')
print(sess.run(tf.matrix_determinant(D)))
print()

E = tf.convert_to_tensor(np.array([ [4.0, 2.0], [-3.0, 5.0] ]))
print('\n Matrix Determinant E:')
print(sess.run(tf.matrix_determinant(E)))
print()

# Matrix Inverse 역행렬 구하기
print(sess.run(tf.matrix_inverse(D)))

# Cholesky Decomposition
print(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors 고유치와 고유 벡터 구하기
print(sess.run(tf.self_adjoint_eig(D)))