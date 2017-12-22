import tensorflow as tf

tw = tf.summary.FileWriter('log_dir', graph=sess.graph)



# 상수 정의하기 - 1
a = tf.constant(14)
b = tf.constant(5)



add = a + b
sub = a - b 

# 세션 실행하기 - 4
sess = tf.Session()


print('더하기:', sess.run(add))
print('빼기:', sess.run(sub))

# f(x) = a * x + b 를 Tensorflow 를 이용하여 실습해보세요.
# 상수 a와 b는 각각 2, 1 이고, 변수 x는 초기값이 5라고 가정하고 풀어보세요.
# 실행결과 : 11
a = tf.constant(2)
b = tf.constant(1)
x = tf.Variable(5)

result = tf.add(tf.multiply(a,x), b)


sess = tf.Session()
sess.run(tf.global_variables_initializer() )

print('a는:', a)
print('a결과 출력:', sess.run(a))
print('실행결과:', sess.run(result))



