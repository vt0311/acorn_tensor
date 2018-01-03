'''
Created on 2018. 1. 3.

@author: acorn
'''

# Operations (쿡북 p41)
#----------------------------------
#
# This function introduces various operations
# in TensorFlow

# Declaring Operations
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# div() vs truediv() vs floordiv()
# div는 매개 변수와 동일한 타입을 반환한다.(소수점 버림)
print(sess.run(tf.div(3,4)))

# truediv는 소수값의 계산 결과를 반환해준다.
print(sess.run(tf.truediv(3,4)))

# 비권장 (0.5 같은 수의 경우 가까운 짝수에 붙는다) 
print(sess.run(tf.floordiv(3.0,4.0)))

# Mod function - 나눗셈의 나머지(자바의 %, 오라클의 mod 함수)
print(sess.run(tf.mod(22.0,5.0)))

# Cross Product - 외적
print(sess.run(tf.cross([1.,0.,0.],[0.,1.,0.])))

# Trig functions (pi : 3.15 - 180도)
print('sin(3.1416):')
print(sess.run(tf.sin(3.1416)))

print(sess.run(tf.cos(3.1416)))

# Tangent = 사인 / 코사인
# 3.1416 / 4.0 은 45도 (탄젠트 45도는 1)
print(sess.run(tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.))))

# Custom operation
test_nums = range(15)

#from tensorflow.python.ops import math_ops
#print(sess.run(tf.equal(test_num, 3)))
def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return(tf.subtract(3 * tf.square(x_val), x_val) + 10)

# 3 * 11^2 - 11 + 10 = 3 * 121 - 11 + 10 = 362  
print('sess.run(custom_polynomial(11)):')
print(sess.run(custom_polynomial(11)))

# What should we get with list comprehension
expected_output = [3*x*x-x+10 for x in test_nums]
print('expected_output:')
print(expected_output)

# TensorFlow custom function output
for num in test_nums:
    print(sess.run(custom_polynomial(num)))
    