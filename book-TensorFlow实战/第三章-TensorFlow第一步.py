#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/24 11:55
# @Author  : HouJP
# @Email   : houjp1992@gmail.com

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# load MNIST dataset
mnist = input_data.read_data_sets("MINIST_data/", one_hot=True)

# show info of MNIST
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# define session
sess = tf.InteractiveSession()

# define placeholder
x = tf.placeholder(tf.float32, [None, 784])  # images
y_ = tf.placeholder(tf.float32, [None, 10])  # labels

# define variable
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# init variables
tf.global_variables_initializer().run()

# mini-batch
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))