import numpy  as np
 
print('\nnp.repeat 함수는 su를 rep_cnt만큼 반복한다.')
su1 = 2
rep_cnt = 5
result = np.repeat(su1, rep_cnt)
print( type(result ))
print( result )
 
array1 = np.array([1, 2])
array2 = np.array([3, 4])
print('\n1번 배열')
print( array1 )
 
print('\n2번 배열')
print( array2 )
 
print('\nnp.concatenate 함수는 배열들을 합쳐 준다.')
result = np.concatenate((array1, array2)) 
print( result )
 
su2 = 3
rep_cnt2 = 4
print('\n함수들의 중첩.')
result = np.concatenate((np.repeat(su1, rep_cnt), np.repeat(su2, rep_cnt2))) 
print( result )
 
array3 = np.array([1, 2, 3, 4, 5, 6])
print('\nreshape 함수는 형상을 변경해준다.')
 
print('2행 3열')
result = np.reshape(array3, [2, 3])
print( result )
 
print('\n3행 2열')
result = np.reshape(array3, [3, 2])
print( result )
 
array4 = np.array([[3, 6, 2], [4, 1, 5]])
print('\n4번 배열')
print( array4 )
 
print('\n전치된 배열')
result = np.transpose(array4)
print( result )