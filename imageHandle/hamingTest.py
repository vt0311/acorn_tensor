# 해밍 거리 테스트

import numpy as np

# 해밍 거리는 같은 수를 가진 2개의 문자열에서 대응하는 위치에 있는 문자 중 다른 것들의 개수 이다.
 
# x와 y의 요소들중에서 서로 다른 것은 3개이다.
x = np.array([1,2,3,4])
y = np.array([1,1,0,3])
 
# 요소 값들이 다른 것들만 합치기
dist = (x != y).sum()

print('dist:', dist) # 3

print('len(x):', len(x) )

print( 'dist / len(x):', dist / len(x) ) 

print( '배열의 유사율 : ', str(100 * (1 - dist / len(x) )), '%' )