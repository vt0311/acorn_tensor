# y_2x+1.py
# 간단한 1차 방정식
import tensorflow as tf
import numpy as np

from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

 
# 단계 1 : 모델을 작성한다.
# 1.1 : 입력될 데이터와 출력될 데이터를 설정한다.
# x : 입력 데이터, y : 출력될 데이터
# 1.0부터 4.5까지 0.5씩 증가시키기
x = np.arange( 1.0, 4.1, 0.5 )
print( 'x = ', x ) # x =  [ 1.   1.5  2.   2.5  3.   3.5  4. ]
 
y = [ 3.1,  4.1,  4.9,  6.1,  6.9,  8.2,  9.1]
print( 'y = ', y ) # y =  [3.1, 4.1, 4.9, 6.1, 6.9, 8.2, 9.1]
 
# 1.2 가중치와 바이어스의 초기 값을 설정한다.
#w = tf.Variable(0.1)
#b = tf.Variable(0.1)
 
# 1.3 가설을 만든다.
#H = w *  x + b
 
# 1.4 cost 함수를 작성한다.
#diff = tf.square(H - y)#오류
#cost = tf.reduce_mean(diff) #모든 샘플의 오류 평균
 
# 1.5 경사 하강법에 의한 최소화 작업을 수행한다.
#learn_rate = 1e-3 #학습율
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
#train = optimizer.minimize(cost)  #학습
 
# 단계 2,3 : Run/update graph and get results
 
# 2.1 세션 객체를 만든다.
#sess = tf.Session() # 세션 객체를 구한다.
 
# 2.2 그래프 내의 글로별 변수들을 초기화한다.
#sess.run(tf.global_variables_initializer())
 
# 2.3 그래프 내의 변수들을 업데이트 하면서 실행한다.
#for step in range(10000):
#    sess.run(train)
#    print('step : %d, cost : %.12f, weight : %f, bias : %f' % \
#        ( step, sess.run(cost), sess.run(w), sess.run(b)))
    
    
#import matplotlib.pyplot as plt

#cost_list = []
#weight_list = []

#plt.plot(cost, 'b')    

#-----------------------------------------------------------------
# 180112 keras 실습

# 모델 객체를 생성한다.
model = Sequential()

'''
# Sequential : 선형으로 만든 layer

# add() 메소드를 이용하여 필요한 연산을 추가

# Dense : NN layer 를 조밀하게 연결시켜 응축해주는 역할
# Dense는 core layer에서 검색 바람.
# input_dim : 입력의 차원
# activation : 활성화 함수 지정
# units : output의 차원수
'''
# add() 메소드를 이용하여 필요한 연산을 추가
# Dense : NN layer 를 조밀하게 연결시켜 응축해주는 역할
# Dense는 core layer에서 검색 바람.
# input_dim : 입력의 차원
# activation : 활성화 함수 지정
# units : output의 차원수
model.add(Dense(1, input_dim=1))

# 옵티마이져 객체 구하기
sgd = optimizers.SGD(lr=0.1) # lr:학습률

# 필요한 정보를 입력하고, 컴파일
# compile 메소드 매개변수
# optimize : 옵티마이져를 저장한다.
# loss : 손실 함수 지정
# metrics : 훈련/테스트 하는 모델이 평가하는 지표들의 목록
# 예시 : metrics = {'output_result':'accuracy'}
# activation : 활성화 함수 
#compile(source, filename, mode, flags, dont_inherit, optimize)
model.compile(optimizer=sgd, loss='mse') # mse : mean square error

# 모델 정보 간략히 보기
model.summary()


# fit : epochs 숫자만큼 훈련시킨다.
# History object를 반환한다.
# x : training data
# y : label data
# batch_size : 정수 또는 None, 기본 값 : 32
# epochs : 정수, 모델을 학습시키기 위한 epoch 숫자
model.fit(x, y, epochs=200)

# predict() 메소드 : 입력 데이터에 대한 출력 예측치를 배열 형태로 생성해준다.
# x : 입력 데이터
# batch_size() : 기본값 32
# verbose : 0 또는 1
predict = model.predict( x=np.array([5.0]) )
print(predict)

# model 바깥으로 정확도나 손실 함수의 값을 가져올 수 있는 방법은?

print(model.get_weights())

# model 관련 메소드 정리
# get_weights() : w의 b의 값을 반환해준다.
# to_json() : 제이슨 형식으로 반환해준다.
# summary() : 간략한 형식으로 결과 보여주기
#             keras.utils.print_summary(model)와 동일 ?

# 모델의 구성 설정 정보들을 사전 형식으로 반환해준다.
print(model.get_config())


# loss의 종류 : losses으로 검색바람.

    
    
    
    
    
    