import tensorflow as tf

# 플레이스 홀더는 템플릿처럼 값을 넣을 공간을 미리 만들어두었다가 실행이 되는 시점에 치환이 되는 변수를 의미한다.
# 변수의 형태만 미리 정해놓는다.

# 배열의 크기를 [None]으로 지정하게 되면, 임의의 갯수 넣을 수 있다.
a = tf.placeholder(tf.int32, [None])
# 배열의 모든 값을 10배하는 연산 정의하기
b = tf.constant(10)

x10_op = a * b 

# 세션 시작하기
sess = tf.Session()

# 플레이스 홀더에 값을 넣어 실행하기 -- (2)
r1 = sess.run(x10_op, feed_dict={a:[1,2,3,4,5]})
print(r1)


r2 = sess.run(x10_op, feed_dict={a: [10,20]})
print(r2)

