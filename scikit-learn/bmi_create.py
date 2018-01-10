'''
Created on 2018. 1. 10.

@author: acorn

서포트 벡터 머신을 사용하기 위하여 샘플용 엑셀 파일을 만들어 본다.
'''

import random

# 무작위로 2만명의 데이터 만들기

# BMI를 계산해서 레이블을 리턴하는 함수
# BMI 공식 : 몸무게 / ( 키 * 키 )
def calc_bmi(h, w):
    bmi = w / (h/100) ** 2
    if bmi < 18.5: return "thin"
    if bmi < 25: return "normal"
    return "fat"

# 출력 파일 준비하기
fp = open("bmi.csv", "w", encoding="utf-8")
fp.write("height,weight,label\n")

# 무작위로 데이터 생성하기
cnt = {"thin":0, "normal":0, "fat":0}

for i in range(20000):
    h = random.randint(120,200) # 120부터 200미만까지
    w = random.randint(35, 80)
    label = calc_bmi(h, w) # 함수 호출
    cnt[label] += 1
    fp.write("{0},{1},{2}\n".format(h, w, label))
    
fp.close()
print("ok,", cnt)    