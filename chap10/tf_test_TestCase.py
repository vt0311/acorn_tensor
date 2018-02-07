import tensorflow as tf


class SquareTest(tf.test.TestCase):
    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            print(x.eval())
            self.assertAllEqual(x.eval(), [4, 9])

if __name__ == '__main__':
    tf.test.main()