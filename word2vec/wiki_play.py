from gensim.models import word2vec
from nltk.translate.ribes_score import position_of_ngram
from numpy.ma.core import negative

model = word2vec.Word2Vec.load('wiki.model')

# 'Python', '파이썬의 유사어 조사
print(model.most_similar(positive=['Python', '파이썬']))
print()

# 아빠 - 남성 + 여성 = 엄마 
# [0] : 가장 연관성이 높을 것을 의미
print(model.most_similar(positive = ['아빠', '여성'], negative=['남성'])[0])
print()

print(model.most_similar(positive = ['아빠', '여성'], negative=['남성'])[1])
print()

print(model.most_similar(positive = ['아빠', '여성'], negative=['남성']))
print()

# 왕자 - 남성 + 여성 = ??
print(model.most_similar(positive = ['왕자', '여성'], negative=['남성']))
print()

print(model.most_similar(positive = ['왕자', '여성'], negative=['남성'])[0])
print()

print(model.most_similar(positive = ['왕자', '여성'], negative=['남성'])[9])
print()


# 서울 - 한국 + 일본 = ??
print(model.most_similar(positive = ['서울', '일본'], negative=['한국'])[0])
print()

# 중국의 수도는 ?
# 서울 - 한국 + 중국 = ??
print(model.most_similar(positive = ['서울', '중국'], negative=['한국'])[0])
print()

# 서울과 맛집을 동시에 검색
print(model.most_similar(positive = ['서울', '맛집'])[0:5])
print()

print("ok")