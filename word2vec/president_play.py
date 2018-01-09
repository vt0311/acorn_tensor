from gensim.models import word2vec

model = word2vec.Word2Vec.load('president.model')

# '국민'이라는 단어의 유사어 찾기
print(model.most_similar(positive=['국민']))

print("ok")