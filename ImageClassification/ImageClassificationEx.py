# 텐서 플로우 이미지 인식 참조
# [1] https://www.tensorflow.org/versions/r0.8/tutorials/image_recognition/index.html#usage-with-python-api
# -*- coding: utf-8 -*-
# Inception-v3 모델을 이용한 Image Classification
# Inception-v3 모델을 다운로드 해서 이미지 인식(추론)을 하는 프로그램을 만들어보자.

# 즉, 입력으로는 이미지가 주어지면 결과 값으로 이미지에 대한 5개의 추론 결과(레이블을)를 출력하는 프로그램을 만들고자 한다.
#
# 이번 챕터에서는 위의 과정을 수행하는 모델을 자세하게 만들지는 않고 우선 구글에서 미리 만들어 놓은 
# Inception-v3라는 모델을 이용해서 이미지 인식(추론) 프로그램이 어떻게 작동하는지 감만 잡아보자.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   모델의 그래프 파일

# imagenet_synset_to_human_label_map.txt:
#   imagenet_synset 파일
#   synset ID를 인간이 읽을수 있는 문자로 맵핑

# imagenet_2012_challenge_label_map_proto.pbtxt:
#   imagenet_synset 파일이라고 한다.
#   protocol buffer의 문자 표현을 synset ID의 레이블로 맵핑

# Inception-v3 모델을 다운로드 받을 경로를 설정
tf.app.flags.DEFINE_string(
    'model_dir', '.\\tmp\\imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")

tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# 다운로드 받을 경로
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

class NodeLookup(object): # object 상속
    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """각각의 softmax node에 대해 인간이 읽을 수 있는 영어 단어를 로드 함.
        Args:
          label_lookup_path: 정수 node ID에 대한 문자 UID.
          uid_lookup_path: 인간이 읽을 수 있는 문자에 대한 문자 UID.
        Returns:
          정수 node ID로부터 인간이 읽을 수 있는 문자에 대한 dict.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
            
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        
        p = re.compile(r'[n\d]*[ \S,]*')
        
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        node_id_to_uid = {}
        
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()  


        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        node_id_to_name = {}
        
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def create_graph():
    """저장된 GraphDef 파일로부터 그래프를 생성하고 저장된 값을 리턴함."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image):    
    # 파일이 없으면 
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    # 파일이 있으면    
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    create_graph()

    with tf.Session() as sess:
        # 몇가지 유용한 텐서들:
        # 'softmax:0': 1000개의 레이블에 대한 정규화된 예측결과값(normalized prediction)을 포함하고 있는 텐서
        # 'pool_3:0': 2048개의 이미지에 대한 float 묘사를 포함하고 있는 next-to-last layer를 포함하고 있는 텐서
        # 'DecodeJpeg/contents:0': 제공된 이미지의 JPEG 인코딩 문자를 포함하고 있는 텐서

        # image_data를 인풋으로 graph에 집어넣고 softmax tesnor를 실행한다.
        # get_tensor_by_name : 주어진 이름을 가진 텐서를 구해준다.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # nodeId --> 영어 단어 lookup을 생성한다.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

def maybe_download_and_extract():
    """Download and extract model tar file."""
    # FLAGS.model_dir : 자바의 Enum과 유사
    dest_directory = FLAGS.model_dir
#     print(dest_directory)

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory) # 없으면 만들고...
       
    # 파일 이름 : 해당 url 끝에 있는 파일 이름과 동일    
    filename = DATA_URL.split('/')[-1]
    
    filepath = os.path.join(dest_directory, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    
    # tar 형식의 압축 파일 열기    
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def main(argv=None):
    maybe_download_and_extract()
    
    # 인풋으로 입력할 이미지를 설정한다.
    # run_inference_on_image(image)
    # run_inference_on_image( 'funny_cat.jpg')
    # run_inference_on_image('mydog.png')
    run_inference_on_image('myrabbit.jpg')

if __name__ == '__main__':
    main()
    