# 배열 인덱싱
# Numpy는 배열을 인덱싱하는 몇 가지 방법을 제공한다.
#
# 슬라이싱: 파이썬 리스트와 유사하게, Numpy 배열도 슬라이싱이 가능하다.
# Numpy 배열은 다차원인 경우가 많기에, 각 차원별로 어떻게 슬라이스할건지 명확히 해야 한다:
import numpy as np
 
# 3행 4열의 이차원 배열(shape (3, 4)) 생성
arrTwoDim = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
 
print('\n # 인덱싱과 슬라이싱을 위한 이차원 배열 출력')
print( 'arrTwoDim:' )
print( arrTwoDim )
print( )
 
# 슬라이싱을 이용하여 첫 두 행과 1열, 2열로 이루어진 부분 배열을 만들어 보도록 한다.
# b는 shape가 (2,2)인 배열이 됩니다:
print('\n # 0~1행 1~2열 까지의 부분 배열 b:')
b = arrTwoDim[:2, 1:3]
print( b )
print( )
 
print('\n# 슬라이싱된 배열은 원본 배열과 같은 데이터를 참조합니다,')
print('# 즉 슬라이싱된 배열을 수정하면 원본 배열 역시 수정됩니다.')
print ('arrTwoDim[0, 1]:', arrTwoDim[0, 1])   # 출력 : 2
print( )
 
b[0, 0] = 77    # b[0, 0]은 a[0, 1]과 같은 데이터입니다
print (arrTwoDim[0, 1])   # 출력 : 77
print( )
 
# 정수를 이용한 인덱싱과 슬라이싱을 혼합하여 사용할 수 있습니다.
# 하지만 이렇게 할 경우, 기존의 배열보다 낮은 rank의 배열이 얻어집니다.
 
# 3행 4열의 이차원 배열(shape (3, 4)) 생성
array6 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print( array6 )
print( )
 
# 배열의 중간 행에 접근하는 두 가지 방법이 있습니다.
# 정수 인덱싱과 슬라이싱을 혼합해서 사용하면 낮은 rank(몇 차원인가?)의 배열이 생성되지만,
row_r1 = array6[1, :] # 1행의 모든 열
 
# 슬라이싱만 사용하면 원본 배열과 동일한 rank의 배열이 생성됩니다.
row_r2 = array6[1:2, :]  # 배열a의 두 번째 행을 rank가 2인 배열로
 
# shape는 각 차원의 크기를 알려주는 정수들이 모인 튜플이다.
print (row_r1, row_r1.shape)  # 출력 "[5 6 7 8] (4,)"
print( )
 
print (row_r2, row_r2.shape)  # 출력 "[[5 6 7 8]] (1, 4)"
print( )

# 행이 아닌 열의 경우에도 마찬가지입니다:
col_r1 = array6[:, 1] # 모든 행의 1열
col_r2 = array6[:, 1:2]
 
print (col_r1, col_r1.shape)  # 출력 "[ 2  6 10] (3,)"
print( )
 
print (col_r2, col_r2.shape)
print( )
