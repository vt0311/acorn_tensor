import numpy as np
 
# x와 y의 요소들중에서 서로 다른 것은 3개이다.
x = np.array([1,2,3,4])
y = np.array([1,1,0,3])
 
# 요소 값들이 다른 것들만 합치기
dist = (x != y).sum()
print( dist / len(x) ) # 3
print( '배열의 유사율 : ', str(100 * (1 - dist / len(x) )), '%' )