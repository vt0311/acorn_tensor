import tensorflow as tf

print('이 예시는 순수 파이썬 코드로만 비용 함수를 구해보는 예시이다.')

x = 65
w = 2
b = 20
y = 165
H = w * x + b
cost = (H - y) ** 2
print('\n w=2, b=20, 몸무게 65kg 이고, 키가 165cm인 사람에 대한 비용 함수.')

print('y=', y, ', H=', H, ', cost=', cost)

x = [65, 80, 90, 45]
y = [165, 190, 160, 185]

def data(input):
    result = []
    for item in range(len(input)):
        result.append(2 * input[item] + 20)
    return result


H = data(x)

print('\n 사람의 정보.')
print('몸무게')
print(x)
print('키')
print(H)

cost = 0
m = len(x)

print('\n 사람 각각의 비용 함수')
for step in range(len(x)):
    cost += (H[step] - y[step]) ** 2
    print( (H[step] - y[step]) ** 2 , end=' ')
    
cost = cost / m
print('\n\n 4 사람에 대한 비용 함수의 평균.')
print('cost=', cost)

