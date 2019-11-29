import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_MIN_LOG_LEVEL'] = '2'


def get_result(input1, reuse=False):
    with tf.variable_scope('one', reuse=reuse):
        weight1 = tf.get_variable(name='weight1', shape=[2, 3], initializer=tf.random_normal_initializer(seed=1))
        bias1 = tf.get_variable(name='bias1', shape=[1, 3], initializer=tf.zeros_initializer())
        layer1 = tf.nn.relu(tf.matmul(input1, weight1) + bias1)
    return layer1


def train():
    input1 = tf.Variable(tf.constant([[1, 2], [2, 3]], shape=[2, 2], dtype=tf.float32), name='input1')
    y = get_result(input1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(y))

    with tf.variable_scope('one', reuse=True):
        weight1 = tf.get_variable(name='weight1', shape=[2, 3])
        print(sess.run(weight1))

    a = tf.constant([1, 2], name='a')
    b = tf.constant([2, 4], name='b')
    c = tf.constant([1, 2], name='c')
    d = tf.constant([2, 4], name='d')
    result = a + b
    sess.run(result)
    print(str(result.name))
    result = c + d
    sess.run(result)
    print(str(result.name))
    # print(tf.global_variables())
