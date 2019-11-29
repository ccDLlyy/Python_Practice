import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


# hyperparameter
INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_SIZE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001

MOVING_AVERAGE_DECAY = 0.99

TRAINING_STEPS = 8000


# forward propagation
def forward_propagation(input_layer, aver_class, weights1, biases1, weights2, biases2):
    if aver_class is None:
        hidden_layer = tf.nn.relu(tf.matmul(input_layer, weights1) + biases1)
        output_layer = tf.matmul(hidden_layer, weights2) + biases2
    else:
        hidden_layer = tf.nn.relu(tf.matmul(input_layer, aver_class.average(weights1)) + aver_class.average(biases1))
        output_layer = tf.matmul(hidden_layer, aver_class.average(weights2)) + aver_class.average(biases2)
    return output_layer


# train
def train(minist):
    # batch
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='input_layer')
    y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='output_layer')

    # parameters
    weights1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_SIZE]))
    weights2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, OUTPUT_SIZE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))

    # iteration steps
    global_step = tf.Variable(0, trainable=False)

    # forward propagation in train set
    y_ = forward_propagation(x, None, weights1, biases1, weights2, biases2)

    # # moving average
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # # forward propagation in validation/development and test set
    average_y_ = forward_propagation(x, variable_averages, weights1, biases1, weights2, biases2)

    # loss = cross_entropy_mean + regularization
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.argmax(y, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights1)
    loss = cross_entropy_mean + regularization

    # learning rate decay
    learning_rate = \
        tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                   minist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    # Gradient Descent
    update_parameter_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # combine two operations
    with tf.control_dependencies([update_parameter_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # evaluate
    correct_prediction = tf.equal(tf.arg_max(average_y_, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        validation_feed = {x: minist.validation.images, y: minist.validation.labels}
        test_feed = {x: minist.test.images, y: minist.test.labels}

        accuracy_curve_validation = []
        validation_accuracy = sess.run(accuracy, feed_dict=validation_feed)
        accuracy_curve_validation.append(validation_accuracy)
        for i in range(1, TRAINING_STEPS + 1):
            x_now, y_now = minist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: x_now, y: y_now})
            if i % 1000 == 0:
                validation_accuracy = sess.run(accuracy, feed_dict=validation_feed)
                accuracy_curve_validation.append(validation_accuracy)
                print('After %d iteration steps, validation accuracy is %g' % (i, validation_accuracy))

        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
        print('\n\nAfter %d iteration steps, test accuracy is %g' % (i, test_accuracy))

        plt.plot(np.arange(0, len(accuracy_curve_validation)), accuracy_curve_validation)
        plt.show()


# entrance
def main(argv=None):
    mnist = input_data.read_data_sets('/home/cclyy/data/mnist/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
