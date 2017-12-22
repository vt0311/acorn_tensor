# y_2x+1_reduce_mean.py
# 이 예시는 # y_2x+1.py에서 나온 첫 번째 결과를
# 직접 파이썬 산술 연산으로 테스트 해본 예시이다.
 
import numpy as np
 
# x : 입력 데이터, y : 출력될 데이터
x = np.arange( 1.0, 4.1, 0.5 )
y = [ 3.1,  4.1,  4.9,  6.1,  6.9,  8.2,  9.1]
 
# 참고로 아래의 w, b는 y_2x+1.py 파일에서 테스트한 첫 번째 출력된 w, b의 값이다.
w = 0.132364
b = 0.111414
 
H = [] # 가설 정보를 담을 리스트
for step in range(len(x)):
    H.append( w * x[step] + b )
 
diff = []
for step in range(len(H)):
    diff.append( (H[step] - y[step]) ** 2 )
 
cost = 0.0 # 비용
for step in range(len(diff)):
    cost += diff[step]
cost = cost / len(diff)
 
print( H )
print()
print( diff )
print()
print( cost )