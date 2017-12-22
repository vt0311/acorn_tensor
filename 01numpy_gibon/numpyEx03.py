import numpy as np
# numpy 문제 03
A = [[15, 5], [0, -5]]
B = [[10, -5], [5, 15]]
# 다음 등식을 만족시키는 행렬 X를 구하시오.
# (1) 5X - 2A = 3X + 4B
#X= [[35, -5], [10, 25]]
#2X = 4B + 2A
#X = 2B + A
result1 = np.add(np.multiply(2,B) , A) 
print( result1 )

# (2) 3(X + 2A) = X + 2(A + B)
# 3X + 6A = X + 2A +2B
# 2X = -4A + 2B
# X = -2A + B
result2 =  np.add(np.multiply(-2,A) , B)
print(result2)