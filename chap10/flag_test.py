import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

FLAGS = flags.FLAGS

def printme():
    print('------------------------------------')
    
    # 아래와 같이 직접적으로 FLAGS를 가져와서 정의하고 
    # 사용하는 것도 가능합니다.
    FLAGS = tf.app.flags.FLAGS
    FLAGS.learning_rate = 0.02
    FLAGS.name = 'test'
     
    print(FLAGS.learning_rate)
    print(FLAGS.name)    

def main(_):
    learn_rate = FLAGS.learning_rate
    print('학습율: ', learn_rate)
    printme()

# Run main module/tf App
if __name__ == "__main__":
    tf.app.run()