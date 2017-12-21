#홍길동이가 구매한 총 금액을 numpy를 이용하여 풀어 주세요
#총 금액 = 300 * 4 + 80 * 3 = 1200 + 240 = 1440원
import numpy as np

arrX = np.array([300, 80])
arrY = np.array([4, 3])


result = np.matmul(arrX, arrY)
print('result:', result )

