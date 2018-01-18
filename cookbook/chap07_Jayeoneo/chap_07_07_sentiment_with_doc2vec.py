# Doc2Vec Model

# 교재 335~
# 카페 : 1716(자연어 처리), 1721번(텍스트 분석)
#     
#     Doc2Vec : Document to Vector
#     단어가 포함된 문서와 단어와의 관계 연관성을 따져 보는 기법 
# 
#    word2Vec 이 확장된 개념으로 봐도 무방
#     
#    예시 )  movie  와    love 의 연관 관계
#     문서가 충분히 길고, 만일 주위에 부정적인 단어가 많이 있다면
#     'movie를 love 하지 않는다'라고 해석한다.
# 
#---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit a Doc2Vec model to get
# Document vectors.  From these document vectors, we will split the
# documents into train/test and use these doc vectors to do sentiment
# analysis on the movie review dataset.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
# 모듈화 : 자주 쓰는 함수들은 별도의 함수에 저장해두었다. 
import cookbook.chap07_Jayeoneo.text_helpers

from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Make a saving directory if it doesn't exist
data_folder_name = 'temp'
# 해당 폴더가 존재하지 않으면 
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name) # 해당 폴더를 생성한다.

# Start a graph session
sess = tf.Session()

# Declare model parameters
batch_size = 500
# 빈도가 높은 단어 7500개까지만 처리하겠다.
vocabulary_size = 7500

generations = 100000 # 총학습 횟수

model_learning_rate = 0.001 # 모델을 위한 학습율

embedding_size = 200   # Word embedding size  단어에 대한 임베딩 크기 
doc_embedding_size = 100   # Document embedding size  # 문서에 대한 임베딩 크기
concatenated_size = embedding_size + doc_embedding_size

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 3       # How many words to consider to the left.

# Add checkpoints to training
# 학습 중간 중간에 체크할 포인트
# 5000번마다 임베딩을 저장시키겠다.
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100  # 100번 마다 비용함수 결과를 프린트하겠다. 

# Declare stop words
# 불용어 정의
stops = stopwords.words('english')
#stops = []

# We pick a few test words for validation.
# 검증을 수행할 단어 목록 : 관심있게 지켜볼 단어 정의
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
# Later we will have to transform these into indices

# Load the movie review data
print('Loading Data')
# load_movie_data() : 파일을 읽어서 해당 영화에 대한 리뷰를 로딩한다. 
texts, target = text_helpers.load_movie_data(data_folder_name)
#texts, target = text_helpers.load_movie_data(data_folder_name)

# Normalize text
# 정규화(소문자 처리, 구두점 제외, 숫자는 빼기, whitespace(띄어쓰기,백슬래시등) 제거)  
print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)

# Texts must contain at least 3 words
# 문서는 최소한 3단어 이상 나와야 한다.
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
texts = [x for x in texts if len(x.split()) > window_size]    
assert(len(target)==len(texts))

# Build our data set and dictionaries
print('Creating Dictionary')

# 단어들과 색인으로 구성된 어휘사전을 만들어 주는 함수이다.
# texts : 사전을 만들기 위한 문장 목록
# vocabulary_size : 사용 빈도가 높은 (vocabulary_size - 1 )개 까지만 챙기겠다.
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))

# 문장 리스트를 색인으로 바꿔주는 함수
# texts : 단어 목록
# word_dictionary : 해당 단어들 각각이 등록되어 있는 사전
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

# Get validation word keys
# 검증 받을 단어들에 대한 인덱스 정보를 담고 있는 변수
valid_examples = [word_dictionary[x] for x in valid_words]    

print('Creating Model')
# Define Embeddings:
# 단어와 문서의 임베딩 값을 저장한다.
# 단어의 임베딩
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# 문서의 임베딩
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

# NCE loss parameters
# 빈도가 너무 적으면 모델의 수렴에 오차가 있을 수 있다.
# 이를 해결하기 위하여 tf에서는 잡음 비용 대비 오차 함수를 제공한다.
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                               stddev=1.0 / np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Create data/target placeholders
# 플레이스 홀더 정의
# x_inputs에 +1의 의미는 문서에 대한 색인이 필요하여 +1을 더해준다.
# 처리할 단어마다 문서의 색인이 추가로 들어가야 한다.
x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1]) # plus 1 for doc index
y_target = tf.placeholder(tf.int32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
# Add together element embeddings in window:
# 단어의 임베딩 값을 더하고, 문서 임베딩 값을 추가해주는 임베딩 함수를 만든다.
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

doc_indices = tf.slice(x_inputs, [0,window_size],[batch_size,1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings,doc_indices)

# concatenate embeddings
final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(doc_embed)])

# Get loss from prediction
# 비용 함수의 최적화를 위한 함수를 작성한다.
# loss : 비용함수
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=final_embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
                                     
# Create optimizer
# optimizer : 최적화 함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)

# Cosine similarity between words
# 검증 단어(관심 단어)와의 코사인 유사도를 구한다.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Create model saving operation
# 나중에 사용하려고 임베딩 값을 저장해둔다.
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

#Add variable initializer.
# 모델 변수 초기화
init = tf.global_variables_initializer()
sess.run(init)

# Run the doc2vec model.
# 학습 시작
print('Starting Training')
loss_vec = []
loss_x_vec = []
for i in range(generations):  # 총 학습 횟수만큼 학습시킨다.
    # 스킵 그램(교재 306쪽) : 대상 단어로부터 주변 단어 예측 하기 
    # generate_batch_data : 스킵 그램을 일괄 작업 해주는 함수 
    # 학습시 반복문에서 지속적으로 호출이 된다.
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,
                                                                  window_size, method='doc2vec')
    feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

    # Run the train step
    # 학습 단계 실행
    sess.run(train_step, feed_dict=feed_dict)

    # Return the loss
    # 비용 계산 하기
    if (i+1) % print_loss_every == 0:  # 100번마다 출력하겠다.
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))
      
    # Validation: Print some random words and top 5 related words
    # 특정 단어에 대한 상위 5개 연관 단어 출력하기
    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5 # number of nearest neighbors  출력과 가장 가까운 단어 갯수(여기서는 5개만 선택하겠다.)
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
            print(log_str) # 출력 결과 (교재 342쪽 참조)
            
    # Save dictionary + embeddings
    # 5000번 마다 임베딩을 저장
    if (i+1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        # 어휘사전을 pickle 형식으로 지정
        # pickle : 객체를 바이트 형식으로 저장하는 기법
        # dump, load 
        with open(os.path.join(data_folder_name,'movie_vocab.pkl'), 'wb') as f: # (wb 안적으면 wt)
            # dump : 파일 f에 덤핑(저장)하기
            pickle.dump(word_dictionary, f)
        
        # Save embeddings
        # 임베딩 저장
        model_checkpoint_path = os.path.join(os.getcwd(),data_folder_name,'doc2vec_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))

# Start logistic model-------------------------
# 로지스틱 회귀를 적용하여 리뷰에 대한 감정을 예측해본다.
max_words = 20  # 리뷰의 최대 단어 깊이
logistic_batch_size = 500 # 일괄 학습 크기

# Split dataset into train and test sets
# Need to keep the indices sorted to keep track of document index
# 데이터를 학습용과 테스트용으로 분리 ( 80 : 20 )
train_indices = np.sort(np.random.choice(len(target), round(0.8*len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Convert texts to lists of indices
# 리뷰의 단어들을 색인 값으로 변환해준다.
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

# Pad/crop movie reviews to specific length
# 리뷰의 길이가 20단어가 되게 만든다.
# 즉, 많으면 잘라내고, 적으면 0을 덧붙인다.
text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])

# Define Logistic placeholders
log_x_inputs = tf.placeholder(tf.int32, shape=[None, max_words + 1]) # plus 1 for doc index
log_y_target = tf.placeholder(tf.int32, shape=[None, 1])

# Define logistic embedding lookup (needed if we have two different batch sizes)
# Add together element embeddings in window:
log_embed = tf.zeros([logistic_batch_size, embedding_size])
for element in range(max_words):
    log_embed += tf.nn.embedding_lookup(embeddings, log_x_inputs[:, element])

log_doc_indices = tf.slice(log_x_inputs, [0,max_words],[logistic_batch_size,1])
log_doc_embed = tf.nn.embedding_lookup(doc_embeddings,log_doc_indices)

# concatenate embeddings
log_final_embed = tf.concat(axis=1, values=[log_embed, tf.squeeze(log_doc_embed)])

# Define model:
# Create variables for logistic regression
# 모델을 정의한다.
# 로지스틱 회귀변수들을 정의한다.
A = tf.Variable(tf.random_normal(shape=[concatenated_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare logistic model (sigmoid in loss function)
# 로지스틱 모델에 시그모이드 함수를 적용한다. 
model_output = tf.add(tf.matmul(log_final_embed, A), b)

# Declare loss function (Cross Entropy loss)
# 교차 엔트로피 함수 적용
logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.cast(log_y_target, tf.float32)))

# Actual Prediction
# 예측과 정확도 함수 정의
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(log_y_target, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
logistic_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
logistic_train_step = logistic_opt.minimize(logistic_loss, var_list=[A, b])

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start Logistic Regression
print('Starting Logistic Doc2Vec Model Training')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size=logistic_batch_size)
    rand_x = text_data_train[rand_index]
    # Append review index at the end of text data
    # 문서 데이터의 끝에 리뷰 색인을 추가시킨다.
    rand_x_doc_indices = train_indices[rand_index]
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])
    
    feed_dict = {log_x_inputs : rand_x, log_y_target : rand_y}
    sess.run(logistic_train_step, feed_dict=feed_dict)
    
    # Only record loss and accuracy every 100 generations
    # 100번마다 비용 함수의 정확도를 기록한다.
    if (i+1)%100==0:
        rand_index_test = np.random.choice(text_data_test.shape[0], size=logistic_batch_size)
        rand_x_test = text_data_test[rand_index_test]
        # Append review index at the end of text data
        # 문서 데이터의 끝에 리뷰 색인을 추가시킨다.
        rand_x_doc_indices_test = test_indices[rand_index_test]
        rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))
        rand_y_test = np.transpose([target_test[rand_index_test]])
        
        test_feed_dict = {log_x_inputs: rand_x_test, log_y_target: rand_y_test}
        
        i_data.append(i+1)

        train_loss_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        train_loss.append(train_loss_temp)
        
        test_loss_temp = sess.run(logistic_loss, feed_dict=test_feed_dict)
        test_loss.append(test_loss_temp)
        
        train_acc_temp = sess.run(accuracy, feed_dict=feed_dict)
        train_acc.append(train_acc_temp)
    
        test_acc_temp = sess.run(accuracy, feed_dict=test_feed_dict)
        test_acc.append(test_acc_temp)
    if (i+1)%500==0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))


# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()