'''
Created on 2017. 12. 26.

@author: acorn
'''
# matrixMul.py
import numpy as np
 
# 행렬의 곱셈 예시이다.
# Numpy에선 행렬의 곱 연산을 위하여 'dot' 함수를 사용한다.
 
A = np.array([[1, 2]]) # 1행 2열
B = np.array([[-2], [3]]) # 2행 1열
C = np.array([[3, 2], [-1, 0]]) # 2행 2열
 
print(A.dot(B)) # (1, 2)*(2, 1) ==> (1, 1)
print() # [[4]]
 
print(A.dot(C)) # (1, 2)*(2, 2) ==> (1, 2)
print() # [[1 2]]
 
print(B.dot(A)) # (2, 1)*(1, 2) ==> (2, 2)
print()
# [[-2 -4]
#  [ 3  6]]
 
print(C.dot(B)) # (2, 2)*(2, 1) ==> (2, 1)
print()
# [[0]
#  [2]]