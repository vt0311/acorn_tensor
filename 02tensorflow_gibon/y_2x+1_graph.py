# 간단한 1차 방정식
# 파일 이름 : y_2x+1_graph.py
 
import numpy as np
import matplotlib.pyplot as plt
 
def step_function( x ) :
    return 2 * x + 1
 
# 1.0부터 4.5까지 0.5씩 증가시키기
x = np.arange( 1.0, 4.1, 0.5 )
y = step_function(x)
 
# x =  [ 1.   1.5  2.   2.5  3.   3.5  4. ]
print( 'x = ', x )
 
# y = [ 3.  4.  5.  6.  7.  8.  9.]
print( 'y = ', y )
 
# 파란 색 그래프를 위한 실험용 데이터
y_answer = [ 3.1,  4.1,  4.9,  6.1,  6.9,  8.2,  9.1]
plt.plot(x, y_answer, marker='o', color='b')
 
# 실제 y 값 = [ 3.  4.  5.  6.  7.  8.  9.]
# 빨간색 : 실제 직선의 방정식 답
plt.plot(x, y, marker='o', color='r')
plt.xlim(0.5, 4.5) # x값의 상하한선
plt.ylim(2.0, 10.0) # y값의 상하한선
plt.show()