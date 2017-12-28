'''
Created on 2017. 12. 27.

@author: acorn
'''
# 1) 로지스틱  테스트를 수행해 보세요.

# 훈련용 파일 : train02.csv
# 테스트 용 파일 : test02.csv
# 
# 파일 이름 : logisticTest02.py

import tensorflow as tf
import numpy as np



testdata = np.loadtxt('./test02.csv', dtype= np.float32, delimiter=',')

data = np.loadtxt('./train02.csv', dtype= np.float32, delimiter=',')
print('data.shape:', data.shape)

table_row = data.shape[0] # 행 갯수
print('table_row:', table_row)
table_col = data.shape[1] # 열 갯수

print('table_col:', table_col) # 열 갯수 

column = table_col - 1  # 입력데이터의 컬럼 갯수

testing_row = 4 # 테스트 용 데이터 셋 개수
training_row = table_row  # 훈련용 데이터 셋 개수
 
print( 'table_row : %d, training_row : %d' % (table_row , training_row))
 
def normalize(input):
    max = np.max(input,axis=0)
    min = np.min(input,axis=0) 
    out = (input - min)/(max-min)  
    # print (min)
    # print (max)
    return out
 
def rev_normalize(somedata, alist) :
    result = np.min( alist ) + somedata * ( np.max(alist) - np.min(alist) )
    return result 
 
x_train = data[ 0:training_row, 0:column ]
#x_train = data[ :, 0:2 ]
y_train = data[ 0:training_row, column-1:(column) ]
#y_train = data[ :, 1: ]
 
x_train = normalize(x_train)
y_train = normalize(y_train) 
 
x_test  = testdata[:, 0:column ]
#x_test  = testdata[:, 0:2 ]
y_test  = testdata[0:training_row, column-1:(column) ]
#y_test  = testdata[:, 1: ]
 
y_test_origin = y_test

x_test = normalize(x_test)
y_test = normalize(y_test)
 
 
# 파일이 총 2열로 되어있다.
# 모든 행의 앞 1개는 입력으로 본다.
#x_data = data[:, 0]
#print('x_data: ', x_data)

# 모든 행의 맨뒤 1개는 출력으로 본다.
#y_data = data[:, column:(column+1)] 

#print( x_data.shape ) # 입력용
#print( y_data.shape ) # 출력용


x = tf.placeholder(tf.float32, shape =[None, column]) # 
y = tf.placeholder(tf.float32, shape=[None, 1] )

w = tf.Variable(tf.random_normal([column, 1], dtype=tf.float32)) # 1행 1열
#w = tf.Variable(tf.ones([column, 1], dtype=tf.float32))
b = tf.Variable(0.0)

# ?행2열 * 2행1열
#H = tf.matmul( x, w) + b
H = tf.sigmoid( tf.matmul( x, w ) + b )
#H =  tf.multiply(x,w)  + b 
 
diff = y * tf.log( H ) + (1-y)* tf.log(1-H)
#diff = tf.square( H - y)
cost = -tf.reduce_mean(diff)

learn_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)

predicted = tf.cast( H > 0.0, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer() )

epoch = 10000
for step in range(epoch):
    _t, _w, _c, _h, _p, _a = sess.run([train, w, cost, H, predicted, accuracy], 
                              feed_dict = {x:x_train, y:y_train})
    if step % (epoch / 10) == 0 :
        print("step: %d, cost : %f, accuracy: %f" % (step, _c, _a))
        #print("accuracy:", _a)
        #print("h:", _h)
        #print("predicted:", _p)
        
    #print('h:', _h)
    
#x_test =     
#result = sess.run(H, feed_dict={x:x_test}) 
##print('input', x_test) 
#print('Close Price predict', result ) # 예측 

#print('Close Price Real', y_test)

h=sess.run(H, feed_dict={x:x_test})
#print('Input\n', x_test)
print('-------------------------------------')
print('why', sess.run(w))


def dataSum():
    # 데이터를 일목 요연하게 보기 위하여 배열들을 합쳐 주는 함수이다.
    totallist = []  # 전체 목록을 담을 리스트
    for i in range(len(y_test)):  # 열의 갯수 만큼 반복
        sublist = []
        sublist.append( y_test[i][0] )
        sublist.append( h[i][0] )
        sublist.append( y_test_origin[i][0] )
        sublist.append(rev_normalize(h, y_test_origin)[i][0])
        totallist.append(sublist)
 
    return totallist


print('-------------------------------------')
print('-------------------------------------')
print('\n실제 값(정규화 데이터)(y_test)\n', y_test)
print('-------------------------------------')
print('\n학습 결과(정규화 데이터)(h)\n',h)
print('-------------------------------------')
print('-------------------------------------')
print('\n실제값\n', y_test_origin)
print('-------------------------------------')
imsi = rev_normalize(h, y_test_origin)
print('\n학습 결과\n', imsi )
print('-------------------------------------')
print('-------------------------------------')
temp = dataSum( )
print( temp )

 