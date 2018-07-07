#!/usr/bin/python3
import tensorflow as tf


def model(x, y, mean):
  loss = (x-mean) * (x-mean) + (y-mean)*(y-mean)
  opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  return loss, opt

graph = tf.Graph()
with graph.as_default():
  # definition
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)
  mean = tf.Variable(0.0, tf.float32)
  loss, opt = model(x, y, mean)
  # model 
  result = mean

  # train 
with tf.Session(graph=graph) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  for step in range(100):
    _, l, r = sess.run([opt, loss, result], feed_dict={
      x:1.0,
      y:2.0
    })
    print(step, l, r)

