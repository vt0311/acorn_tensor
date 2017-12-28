'''
Created on 2017. 12. 27.

@author: acorn
'''
# softMaxPreTest05.py
# tensorflow의 one hot 사용 예시이다.

import tensorflow as tf

def getCategory( datalist ):
    mylist = ['강아지', '고양이', '토끼', '이구아나', '뭐였지']
 
    for item in range(len(datalist)):
        print(datalist[item], mylist[(int)(datalist[item])])
 
 
y = [3, 0, 4]

nb_classes = 5
# 각 요소를 token 갯수만큼 쪼갠 후 해당 인덱스만 숫자 1로...

onehot = tf.one_hot(y, nb_classes)


sess=tf.Session()

_one = sess.run(onehot)

print('-------------------------')
print(_one)
print()


result = tf.argmax( _one, axis = 1)

print('-------------------------')

result = sess.run(result)
print(result)

print('-------------------------')
