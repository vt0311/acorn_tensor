'''
Created on 2017. 12. 26.

@author: acorn
'''
lists = ['강감찬', '김유신', '김말똥', '이순신', '이성계', '임하룡', '김유신']

# 인덱싱 : 인덱스를 이용하여 어떤 요소를 추출하는 기능
print('4번째 요소 출력')
print(lists[4])
print()
print('#마이너스인 경우 뒤쪽부터 카운터하고, 인덱스는 1부터 시작한다.')
print(lists[-2])
print()

# 슬라이싱 : 인덱스는 0부터 시작한다.
print('1번째부터 2번째 요소까지 슬라이싱')
print(lists[1:3])
print()

print('4번째 요소부터 끝까지 출력')
print(lists[4:])
print()

print('2번째 요소까지 출력')
print(lists[:3])
print()

print('요소 모두 출력')
print(lists[:])
print()

