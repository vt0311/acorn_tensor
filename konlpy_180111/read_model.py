'''
Created on 2018. 1. 11.

@author: acorn

• Word2vec 모델을 읽어 들이고 간단히 살펴 보도록 한다.
• 파일 이름 : read_model.py

'''

from gensim.models import word2vec
model = word2vec.Word2Vec.load('president.model')

print( 'model:' , model)
print()

print( 'type(model):', type(model) )
print()
# Word2Vec(vocab=355, size=200, alpha=0.025)

# 모델 파일을 읽어 들이면 여러 단어를 추출해 볼 수 있다.
# most_similar : 유사한 단어를 확인하고자 할 때 사용한다.
result = model.most_similar(positive=['경제'])

# 이 결과를 이용하여 막대 그래프를 그려 보면 좋을 듯 하다.
print( result )

result = model.most_similar(positive=['취업'])

print( result )