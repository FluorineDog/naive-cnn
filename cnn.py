import tensorflow as tf 
def network(raw_x, y_, dropout_rate):
  x = tf.reshape(raw_x, [-1, 28, 28, 1])
  # apply 64*5*5 kernel, with relu as activation
  # to bs*28*28*14
  conv_layer1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu,
  )

  # apply 2*2 pooling, nonoverlaped
  # to bs*14*14*128
  pool_layer1 = tf.layers.max_pooling2d(
    inputs=conv_layer1, 
    pool_size=[2, 2], 
    strides=2
  )

  # apply 128*5*5 kernel, with relu as activation, 
  # to bs*14*14*128
  conv_layer2 = tf.layers.conv2d(
    inputs=pool_layer1,
    filters=64,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu,
  )

  # apply 2*2 pooling, non-overlaped
  # to bs*7*7*128
  pool_layer2 = tf.layers.max_pooling2d(
    inputs=conv_layer2, 
    pool_size=[2, 2],
    strides=2,
  )

  # to bs*(7*7*64)
  trans = tf.reshape(pool_layer2, [-1, 7*7*64])

  # to bs*1024
  dense_layer= tf.layers.dense(
    inputs=trans,
    units=256,
    activation=tf.nn.relu,
  )

  # add dropout
  # still bs*1024
  dropout_layer=tf.layers.dropout(
    inputs=dense_layer,
    rate=dropout_rate
  )

  # linear to bs*10
  y = tf.layers.dense(
    inputs=dropout_layer, 
    units=10
  )

  # softmax with cross entropy
  loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

  return y, loss

def model(x, y_, dropout_rate):
  y, loss = network(x, y_, dropout_rate)
  # optimize all thing into one
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
  return y, loss, train_step

