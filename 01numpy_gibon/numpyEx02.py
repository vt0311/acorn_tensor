import numpy as np
# numpy 문제 02
# 다음과 같은 배열이 있다.
a = [-1, 3, 2, -6]
b = [3, 6, 1, 2]
# 
# 1) 항목 a를 shape를 2*2 형식으로 변경하여 A를 배열을 만드시오.
arrA = np.array([[-1, 3], [2, -6]])
print('행렬A:') 
print(arrA)
# 2) 항목 b를 shape를 2*2 형식으로 변경하여 B를 배열을 만드시오.
arrB = np.array([[3, 6], [1, 2]])
print('행렬B:') 
print(arrB)
# 3) A와 B에 대하여 다음 문제를 풀어 주세요.
# 3-1) AB
AB = np.matmul(arrA, arrB)
print('AB:')
print(AB)
# 3-2) BA
BA = np.matmul(arrB, arrA)
print('BA:')
print(BA)
# 4-1) b를 전치시켜 b2 배열을 만드시오.
b2 = np.transpose(b)
print('b2:')
print(b2)

# 4-2) a와 b2의 행렬 연산을 수행하면 그 결과는 어떻게 되나?
result = np.matmul(a, b2)
print('result:')
print(result)
