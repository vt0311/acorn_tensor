import numpy  as np
 
array = np.array([1.57, 2.48, 3.93, 4.33])
print('\n배열 출력하기')
print( array )
 
print('\nnp.ceil() 함수는 소수 자리를 올림한다')
result = np.ceil(array)
print( result )
 
print('\nnp.floor() 함수는 소수 자리를 버린다.')
result = np.floor(array)
print( result )
 
print('\nnp.round() 함수는 소수 자리를 반올림한다.')
result = np.round(array)
print( result )
 
print('\n소수점 1자리에서 반올림.')
result = np.round(array, 1)
print( result )
 
print('\n루트 씌우기.')
result = np.sqrt(array)
print( result )
 
array1 = np.array([-1.1, 2.2, 3.3, 4.4])
print('\n배열 출력하기1')
print( array1 )
 
array2 = np.array([1.1, 2.2, 3.3, 4.4])
print('\n배열 출력하기2')
print( array2 )
 
print('\n절대값 구하기.')
result = np.abs(array1)
print( result )
 
print('\n배열 요소들의 값 합치기.')
result = np.sum(array1)
print( result )
 
print('\n배열 요소들의 값 비교하기.')
result = np.equal(array1, array2)
print( result )
 
print('\nnp.sum과 np.equal을 동시에 사용하기')
print('True는 1로 False는 0으로 카운트된다.')
result = np.sum(np.equal(array1, array2))
print( result )
 
print('\n평균 구하기.')
result = np.mean(array2)
print( result )
 
# 배열 연산의 기본적인 수학 함수
arrX = np.array([[1,2],[3,4]], dtype=np.float64)
arrY = np.array([[5,6],[7,8]], dtype=np.float64)
 
print('\n요소별 더하기')
print( arrX + arrY )
print( np.add(arrX, arrY) )
 
print('\n요소별 빼기')
print( arrX - arrY )
print( np.subtract(arrX, arrY) )
 
print('\n요소별 곱하기') 
print( arrX * arrY )
print( np.multiply(arrX, arrY) )
 
print('\n요소별 나누기') 
print( arrX / arrY )
print( np.divide(arrX, arrY) )
 