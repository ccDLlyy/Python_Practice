import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def train():
    # data product
    dataset_size = 128
    rdm = np.random.RandomState()
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

    # forward_propagation
    input__size = 2
    hidden_size = 3
    output_size = 1
    x = tf.placeholder(tf.float32, shape=(None, input__size), name='batch_input')
    y = tf.placeholder(tf.float32, shape=(None, output_size), name='batch_output')
    w1 = tf.Variable(tf.random_normal([input__size, hidden_size], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=1, seed=1))
    a1 = tf.matmul(x, w1)
    y_ = tf.matmul(a1, w2)
    y_ = tf.sigmoid(y_)
    batch_loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_, 1e-10, 1)) + (1-y)*tf.log(1 - tf.clip_by_value(y_, 1e-10, 1)))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(batch_loss)

    # train
    loss_list = []
    with tf.Session() as sess:
        # initialization
        init = tf.global_variables_initializer()
        sess.run(init)

        # training
        print('start training!')
        batch_size = 8
        epoches = 200
        for i in range(epoches):
            j = 0
            while True:
                start = j * batch_size
                end = min(start + batch_size - 1, dataset_size)
                j += 1
                if start >= end:
                    loss = sess.run(batch_loss, feed_dict={x : X, y : Y})
                    loss_list.append(loss)
                    print(str(i) + ' epoch done!')
                    break
                sess.run(optimizer, feed_dict={x: X[start:end], y: Y[start:end]})

    # learning curve
    plt.plot([x + 1 for x in range(len(loss_list))], loss_list)
    plt.show()


if __name__ == '__main__':
    train()
