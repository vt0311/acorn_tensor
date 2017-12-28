'''
Created on 2017. 12. 27.

@author: acorn
'''
# softMaxPreTest03.py
# tensorflow의 one hot 사용 예시이다.

import tensorflow as tf
#import numpy as np
 
y = [0, 1, 2]

nb_classes = 4
# 각 요소를 token 갯수만큼 쪼갠 후 해당 인덱스만 숫자 1로...

onehot = tf.one_hot(y, nb_classes)

print(onehot)
#출력결과
#Tensor("one_hot:0", shape=(3, 4), dtype=float32)

sess=tf.Session()

print(sess.run(onehot))
#출력결과
#[[ 1.  0.  0.  0.] ==> 숫자 1을 토큰 4개로 만든 후 0번째 인덱스만 1로 설정
# [ 0.  1.  0.  0.] ==> 숫자 1을 토큰 4개로 만든 후 1번째 인덱스만 1로 설정
# [ 0.  0.  1.  0.]] ==> 숫자 1을 토큰 4개로 만든 후 2번째 인덱스만 1로 설정

###########################################################################


