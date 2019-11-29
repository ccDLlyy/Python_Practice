import os
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


TIMESTEPS = 10
HIDDEN_SIZE = 30
NUM_LAYERS = 2

TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_SIZE = 10000
TESTING_SIZE = 1000
SAMPLE_GAP = 0.01


def generate_date(seq):
    x = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        x.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


# 定义模型
def lstm_model(x, y, istraining, reuse):
    global_steps = tf.Variable(0, trainable=False)
    with tf.variable_scope('RNN', reuse=reuse):
        ceil = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        # lstm = tf.keras.layers.RNN(ceil)
        # outputs, state = lstm(x, initial_state=state)
        # outputs, state = tf.keras.layers.RNN(ceil, x)
        outputs, state = tf.nn.dynamic_rnn(ceil, x, dtype=tf.float32)
        output = outputs[:, -1, :]
        predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    if not istraining:
        return predictions, None, None
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, global_step=global_steps)
    return predictions, loss, train_op


# 训练
def train(sess, x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    x, y = ds.make_one_shot_iterator().get_next()

    # with tf.variable_scope('model'):
    predictions, loss, train_op = lstm_model(x, y, True, False)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print('train step：' + str(i) + '，loss：' + str(l))


# 验证
def run_eval(sess, x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.repeat().shuffle(1000).batch(1)
    x, y = ds.make_one_shot_iterator().get_next()

    # with tf.variable_scope('model', reuse=True):
    prediction, _, _ = lstm_model(x, [0.0], False, True)
    predictions = []
    labels = []
    for i in range(TESTING_SIZE):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt((predictions - labels) ** 2).mean(axis=0)
    print('rmse' + str(rmse))

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real')
    plt.legend()
    plt.show()


test_start = (TRAINING_SIZE + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_SIZE + TIMESTEPS) * SAMPLE_GAP
train_x, train_y = generate_date(np.sin(np.linspace(0, test_start, TRAINING_SIZE + TIMESTEPS, dtype=np.float32)))
test_x, test_y = generate_date(np.sin(np.linspace(test_start, test_end, TESTING_SIZE + TIMESTEPS, dtype=np.float32)))
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
with tf.Session() as sess:
     train(sess, train_x, train_y)
     run_eval(sess, test_x, test_y)
     writer = tf.summary.FileWriter('./loog', tf.get_default_graph())
     writer.close()