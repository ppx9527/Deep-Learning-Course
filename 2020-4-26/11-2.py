import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd

# 二元分类
train_url = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)

df_data = pd.read_csv(train_path)
x_train, y_train = df_data.values[:, 0:4], df_data.values[:, 4:]
x_train = tf.cast(tf.expand_dims(x_train, axis=1), tf.float32)
x_train, y_train = x_train[y_train < 2], tf.cast(tf.reshape(y_train[y_train < 2], (-1, 1)), tf.float32)

w = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.zeros(1), dtype=tf.float32)

epochs = 50
learning_rate = 0.001

for i in range(epochs):
    with tf.GradientTape() as tape:
        pre = 1/(1 + tf.exp(-(x_train @ w + b)))
        loss = -tf.reduce_sum(y_train * tf.math.log(pre) + (1 - y_train) * tf.math.log(1 - pre))

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.where(pre.numpy() < 0.5, 0.0, 1.0), y_train), tf.float32))
    delta = tape.gradient(loss, [w, b])
    w.assign_sub(delta[0] * learning_rate), b.assign_sub(delta[1] * learning_rate)

    print("epoch：{}，loss：{}，accuracy：{}".format(i, loss, acc))

print(tf.reshape(tf.where(pre.numpy() < 0.5, 0, 1), (-1,)))
print(tf.cast(tf.reshape(y_train, (-1,)), tf.int32))
