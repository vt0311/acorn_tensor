'''
Created on 2018. 1. 10.

@author: acorn

파일 이름 : xor_train_01.py
사이킷 런을 이용하여 XOR 연산을 수행해본다.
'''

from sklearn import svm

# XOR의 계산 결과 데이터 --- (※1)
#P, Q, result
xor_data = [ # 입력과 출력을 한 꺼번에 작성
[0, 0, 0],
[0, 1, 1],
[1, 0, 1],
[1, 1, 0]
]

# 학습을 위해 데이터와 레이블 분리하기 --- (※2)
x = [] # 입력 데이터
y = [] # 출력 데이터
for row in xor_data:
    prediction = row[0]
    q = row[1]
    r = row[2]
    x.append([prediction, q])
    y.append( r )

# 데이터 학습시키기 --- (※3)
# svm 알고리즘을 이용하여 머신 러닝 수행
#clf = svm.SVC()

# max_iter=1 -> 학습률이 50%
#clf = svm.SVC(max_iter=1)

# max_iter=2 -> 학습률이 100%
#clf = svm.SVC(max_iter=2)

#max_iter=1000000 -> 학습률이 100%
clf = svm.SVC(max_iter=1000000)



# fit(학습용_데이터, 레이블_데이터) 함수 : 데이터를 학습시킨다.
clf.fit(x, y)

# 데이터 예측하기 --- (※4)
# predict(예측하고자_하는_데이터)
pre = clf.predict(x)


print("예측 결과:", pre)
# 결과 확인하기 --- (※5)
ok = 0; total = 0
for idx, answer in enumerate(y):
    prediction = pre[idx]
    if prediction == answer: ok += 1
    total += 1

print("정답률:", ok, "/", total, "=", ok/total)




