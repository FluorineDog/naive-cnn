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
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    for step in range(30):
      print("step", step)
      for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        opt, l = sess.run([train_step, loss], feed_dict={
            x: batch_xs, y_: batch_ys, dropout_rate: 0.4})
        if i == 499:
          print("train acc & loss: ", sess.run(
              [accuracy, loss], feed_dict={
                  x: batch_xs,
                  y_: batch_ys,
                  dropout_rate: 1.0
              }
          ))

      tx, ty = mnist.test.next_batch(1000)
      print("partial test acc & loss: ", sess.run(
          [accuracy, loss], feed_dict={
              x: tx,
              y_: ty,
              dropout_rate: 1.0
          }
      ))

  # final with all test
  # use dirty workaround 
  # since the complete test data 
  # cannot fit into GPU
  block_count = 10
  total_acc = 0
  total_loss = 0
  for i in range(block_count):
    tx, ty = mnist.test.next_batch(1000, shuffle=False)
    acc, loss = sess.run(
        [accuracy, loss], feed_dict={
            x: tx,
            y_: ty,
            dropout_rate: 1.0
        }
    )
    total_acc += acc
    total_loss += loss
  final_acc = total_acc / block_count 
  final_loss = total_loss / block_count 
  print("final test acc & loss: ", [final_acc, final_loss])

main()
