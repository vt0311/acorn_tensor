import numpy  as np
 
print('\n임의의 값으로 채워진 2 * 2 배열 생성')
result = np.random.random((2, 2))
print( result )
# [[ 0.76769     0.49427126]
#  [ 0.84546145  0.4961498 ]]
 
print('\n표준 편차가 1이고 평균 값이 0인 정규 분포(matlab과 같은 방식)에서 표본을 추출한다.')
result = np.random.randn(2, 3)
print( result )

 
print('\n임의의 값으로 채워진 4 * 4 배열 생성')
result = np.random.rand(4, 4)
print( result )
 
print('\n임의의 값으로 채워진 3행 3열 배열 생성')
result = np.random.uniform(size=(3, 3))
print( result )
 
print('\n임의의 값으로 채워진 1행 2열 배열 생성')
x_shape = [1, 2] 
result = np.random.uniform(size=x_shape)
print( result )
 
print('\n임의의 값으로 채워진 3면 2행 2열 배열 생성')
x_shape = [3, 2, 2] 
result = np.random.uniform(size=x_shape)
print( result )
 
print('\n0이상 5미만의 임의의 정수 1개를 추출한다.')
result = np.random.randint(5)
print( result )
 
print('\n0이상 3미만의 임의의 정수 4개를 추출한다.')
result = np.random.randint(3, size=4)
print( result )
 
print('\n0이상 5미만의 임의의 정수 10개를 추출한다.')
result = np.random.randint(0, 5, size=10)
print( result )
 
hap = 0
for idx in range(0, 3):
    hap += np.random.randint(3)
 
print( '\n합 : ', end='' ) 
print( hap )
 
# np.random.seed( 12345 )
 
print('\npermutation()은 1~5까지의 숫자를 랜덤하게 섞어 준다.')
length = 5 
result = np.random.permutation( length )
print( result ) 
#[4 3 2 0 1]
 
print('\nnp.random.normal(평균, 표준편차, 요소갯수)')
result = np.random.normal(0, 0.01, 10)
print( result )
 
print('\n0이상 5미만의 정수 중에서 1개 추출)')
result = np.random.choice(5)
print( result )
 
print('\n0이상 5미만의 정수 중에서 3개 추출)')
#This is equivalent to np.random.randint(0,5,3)
result = np.random.choice(5, 3)
print( result )
 