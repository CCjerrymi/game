# forst_tensorflow.py

import tensorflow.examples.tutorials.mnist.input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵
coross_entropy = -tf.reduce_sum(y_*tf.log(y))

