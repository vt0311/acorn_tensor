'''
Created on 2017. 12. 28.

@author: acorn
'''

# 모듈을 임포트 한다.
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
 
# read_data_sets 함수를 이용하여 하드 디스크에 다운로드한다.
# MNIST_data : 디렉토리가 생성된다.
# one_hot=True 옵션은 y의 값을 자동으로 one hot 처리해주는 옵션이다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
# 바로 위의 문장의 mnist 객체에는 다음과 같은 데이터가 들어 있다.
# mnist.train : 훈련용 데이터
# mnist.test : 테스트 데이터
# 훈련용/테스트용 모두 xs(이미지), ys(레이블)를 가지고 있다.
# mnist의 타입 : 
# <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
 
img_row = 28 # 이미지 1개의 가로 픽셀 수
img_column = 28 # 이미지 1개의 세로 픽셀 수
 
# next_batch 함수는 전체 데이터 중에서 subset을 가져 오는 함수이다.
# 메모리의 용량 때문에 조금씩 가져 온다.
batch_xs, batch_ys = mnist.train.next_batch(1) # 1개씩 가져 오기
 
# numpy의 ndarray 타입이다.
print(type(batch_xs)) # <class 'numpy.ndarray'>
 
# 각 이미지 1개는 28 * 28 픽셀로 구성이 되어 있다.
print(batch_xs.reshape( img_row, img_column ))
 
print(batch_xs.shape) # shape(1, 784)
 
# batch_ys가 [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]]이라면
# 숫자 9이다.
print(batch_ys) #  레이블 보기
 
# batch_ys.shape는 출력될 결과물의 형상 정보이다.
# 0부터 9까지의 숫자이어야 하므로 shape(1, 10)이다.
print(batch_ys.shape)
 
plt.imshow(batch_xs.reshape( img_row, img_column ), cmap='Greys')
plt.show()
