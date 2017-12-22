import tensorflow as tf

# 상수 정의하기 - 1
a = tf.constant(120, name='a')
b = tf.constant(130, name='b')

# 변수 정의 및 초기 값 할당 - 2
v = tf.Variable(0, name='v')

# 데이터 플로우 그래프 정의하기 - 3
result = a + b 

# 세션 실행하기 - 4
sess = tf.Session()

# v의 내용 출력하기 -- 5 
print('결과:', sess.run(result))
# 결과 : 250

# 추가문제
# 2개의 정수에 대한 가감승제 연산을 해보세요.
result2 = a - b
print('결과2:', sess.run(result2))

result3 = a * b
print('결과3:', sess.run(result3))

result4 = a / b
print('결과4:', sess.run(result4))






