#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import cnn


FLAGS = None
from tensorflow.examples.tutorials.mnist import input_data


def main(*argv):
  data_dir = "/tmp/tensorflow/mnist/input_data"
  # load dataset
  mnist = input_data.read_data_sets(data_dir)

  graph = tf.Graph()
  with graph.as_default():
    # import data as placeholder
    # batch_size * height * width * channels
    # as bs*28*28*1
    x = tf.placeholder(tf.float32, [None, 784])
    # true label
    y_ = tf.placeholder(tf.int64, [None])
    dropout_rate = tf.placeholder(tf.float32)
    y, loss, train_step = cnn.model(x, y_, dropout_rate)

  with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # Train
    for _ in range(100):
      for _ in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
                 x: batch_xs, y_: batch_ys, dropout_rate: 0.4})

      # evaluate train model
      correct_prediction = tf.equal(tf.argmax(y, 1), y_)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(
          [accuracy, loss], feed_dict={
              x: mnist.test.images,
              y_: mnist.test.labels,
              dropout_rate: 1.0
          }
      ))

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(
    #     accuracy, feed_dict={
    #         x: mnist.test.images,
    #         y_: mnist.test.labels
    #         dropout_rate: 1.0
    #     }
    # ))


# def run():
#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       '--data_dir',
#       type=str,
#       default='/tmp/tensorflow/mnist/input_data',
#       help='Directory for storing input data')
#   FLAGS, unparsed = parser.parse_known_args()
#   FLAGS.dropout_rate = 0.4
#   print(FLAGS)
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# if __name__ == '__main__':
#   run()


main()
