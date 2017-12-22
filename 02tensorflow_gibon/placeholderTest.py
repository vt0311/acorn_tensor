import tensorflow as tf

# 플레이스 홀더는 템플릿처럼 값을 넣을 공간을 미리 만들어두었다가 실행이 되는 시점에 치환이 되는 변수를 의미한다.
# 변수의 형태만 미리 정해놓는다.

# 배열의 크기를 [None]으로 지정하게 되면, 임의의 갯수 넣을 수 있다.
#x1 = tf.placeholder(tf.int32, [None])
# 배열의 모든 값을 10배하는 연산 정의하기
#x2 = tf.placeholder(tf.int32, [None])

#x10_op = a * b 

# 세션 시작하기
#sess = tf.Session()

# 플레이스 홀더에 값을 넣어 실행하기 -- (2)
#r1 = sess.run(x10_op, feed_dict={a:[1,2,3,4,5]})
#print(r1)

#r2 = sess.run(x10_op, feed_dict={a: [10,20]})
#print(r2)
#######################################################################
#수식 : result = 2 * x1 + 3 * x2 + 1
# 위 수식을 처리해주는 placeholder를 구현하시오.

#x1 = [3, 5, 4]
#x2 = [2, 3, 5]
#result = [?, ?, ?]
#result = [13, 20, 24]

result = tf.placeholder(tf.int32, [None])
x1 = tf.placeholder(tf.int32, [None])
x2 = tf.placeholder(tf.int32, [None])
a1= tf.constant(2)
b1= tf.constant(3)
c1 = tf.constant(1)
result =  a1 * x1 + b1 * x2 + c1

# 세션 시작하기
sess = tf.Session()

r1 = sess.run(result, feed_dict={x1: [3, 5, 4] , x2: [2, 3, 5] })
print(r1)
#r2 = sess.run(result, feed_dict={x2: [2, 3, 5]})
#print(r2)