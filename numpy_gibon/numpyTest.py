import numpy as np
 
print('\n배열 출력')
arrA = np.array([[4, 5], [3, 4]])
print( arrA )
 
print('\nnp.matrix 함수는 행렬로 변환해분다.')
x_matrix = np.matrix(arrA)
print( x_matrix )
 
print('\n전치 행렬 출력')
x_transpose = np.transpose(x_matrix)
print( x_transpose )
 
print('\n배열1 출력')
arrA = np.array([[1, 2], [3, 4]])
print( arrA )
 
print('\n배열2 출력')
arrB = np.array([[1, 0], [2, 1]])
print( arrB )
 
print('\nnp.matmul 함수는 행렬 연산')
result = np.matmul(arrA, arrB)
print( result )
 
print('\n배열3 출력')
arrC = np.array([[1, 2], [3, 4]])
print( arrC )
 
print('\nnan이 들어 있는 배열4 출력')
arrD = np.array([[np.nan], [2]])
print( arrD )
 
print('\nnan은 0으로 최환됨')
arrD = np.nan_to_num(arrD)
print( arrD )
 
newdata = [1 for i in range(2)]
 
print('\n배열4 New 출력')
print('column_stack 함수는 모든 행에 대하여 열을 1개씩 추가해준다.')
newArrD = np.column_stack((arrD, newdata))
print( newArrD )
 
print('\nnp.matmul 함수는 행렬 연산')
result = np.matmul(arrC, newArrD)
print( result )
 
print('\nnp.arange는 파이썬의 range와 동일한 개념이다.')
print('0부터 1씩 증가하여 5직전까지 출력하기.')
result = np.arange(5)
print( result )
 
print('1부터 1씩 증가하여 5직전까지 출력하기.')
result = np.arange(1, 5)
print( result )
 
print('1부터 3씩 증가하여 10직전까지 출력하기.')
result = np.arange(1, 10, 3)
print( result )