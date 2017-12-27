'''
Created on 2017. 12. 27.

@author: acorn
'''
# 참치와 꽁치(로지스틱)
# 로지스틱 회귀 모델은 이항 분류 모델이다.
# 어떠한 분류를 해보니 종류가 2개만 나타나는 분류를 이항 분류(binary classfication)라고 한다.
# 
# 생선의 종류를 알려주는 자료가 다음과 같이 있다고 가정하자.
# 이 파일은 참치(분류 : 0)와 꽁치(분류 : 1)를 구분하기 위한 용도로 사용되는 자료이다.
# 이것을 이용하여 테스트용 데이터에 대한 검증을 수행해보세요.
# 
# 파일 이름 : LogisticRegressionEx01.py
# 
# 참조 문서 : Logistic Regression
# 
# 아래 표를 보면 길이가 50cm이고, 무게가 15kg인 고기의 분류는 0(참치)이다.

# LogisticRegressionEx01.py
# Logistic Regression : 입력에 따른 정답이 0또는 1 중의 하나의 값으로 떨어질 때 사용하는 기법
import tensorflow as tf
import numpy as np
 
def normalize(input):
    # min-max 정규화 알고리즘
    max = np.max(input,axis=0)
    min = np.min(input,axis=0)
    out = (input - min)/(max-min)
    return out
 
x_data = [[50,15], [40,20], [10,5], [20,10]]
y_data = [[0],     [0],     [1],    [1]]
 
x_data = normalize( x_data )
y_data = normalize( y_data )
 
x_test = [[45,22], [15,13]]
y_test = [[0],     [1]    ]
x_test = normalize( x_test )
y_test = normalize( y_test )
 
column = 2
x = tf.placeholder(tf.float32,[None, column]) # 4행 2열
y = tf.placeholder(tf.float32,[None, 1]) # 4행 1열
 
w = tf.Variable(tf.ones([column, 1]),tf.float32) # 2행 1열
b = tf.Variable(0.0)
 
# (4행 2열) * (2행 1열) ==> (4행 1열)
H = tf.sigmoid( tf.matmul( x, w ) + b )
 
# 로지스틱 회귀에서 차이를 구하는 공식은 다음과 같다.
diff = y * tf.log(H) + (1 - y) * tf.log(1 - H)
cost = -tf.reduce_mean( diff )
 
learn_rate = 0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate = learn_rate )
train = optimizer.minimize( cost )
 
# H(가설)이 0.5이상이면 1.0으로, 아니면 0.0으로 변경시킨다.
predicted = tf.cast( H  > 0.5, dtype=tf.float32)
 
# 예측한 값과 실제 y값이 동일하면 1.0, 그렇지 않으면 0.0으로 만든 다음 평균 값을 구한다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
 
sess=tf.Session()
sess.run(tf.global_variables_initializer())
 
def inlineprint( mylist ):
    imsi = ''
    for item in mylist :
        imsi += str(item) + ' '
    print (imsi)
 
for step in range(10000):
    _t, _w, _c, _h, _p, _a  = sess.run([train, w, cost, H, predicted, accuracy],
                      feed_dict = {x : x_data, y : y_data } )
    if step % 1000 == 0 :
        print('학습 회수 : %d, 비용 : %f, 정확도 : %f' % (step, _c, _a))
        print('가설', end = ' : ')
        inlineprint( _h )
        print('예측 결과', end=' : ')
        inlineprint( _p )
        print('-----------------------------------------------------------------------------')
 
predict = sess.run(predicted, feed_dict = { x : x_test, y : y_test })

#print('테스트 데이터의 예측 결과')

print('테스트 데이터의 예측 결과', end=' : ')

inlineprint( predict )

def getCategory(datalist):
    mylist = ['참치', '꽁치']
    for item in range(len(datalist)):
        print( datalist[item], mylist[ (int)(datalist[item]) ] )
        
getCategory(predict)
        
         
