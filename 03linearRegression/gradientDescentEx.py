'''
Created on 2017. 12. 26.

@author: acorn
'''
# GradientDescentEx.py
# 경사 하강법에 사용되는 공식을 파이썬에서 검증해보는 예시이다.
x = 1.0
y = 1.0
 
alpha = 0.01 # 학습율
m = 1 # 도수
 
# w = 5.0 이라고 가정하고 다음 값이 어떻게 바뀌는 지 살펴 본다
w = 5.0
H = w * x
 
print( 'w :', w)
def calc( w ) :
    result = w - alpha * (1 / m) * (w * x - y) * x
    return result
 
for step in range(2) :
    w = calc( w )
    print('w :', w)