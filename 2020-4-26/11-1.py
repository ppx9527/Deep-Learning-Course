import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd

# 多元分类
train_url = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)

test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

df_data = pd.read_csv(train_path)
x_train, y_train = df_data.values[:, 0:4], df_data.values[:, 4:]
df_data = pd.read_csv(test_path)
x_test, y_test = df_data.values[:, 0:4], df_data.values[:, 4:]
y_train = tf.squeeze(tf.one_hot(tf.cast(y_train, tf.int32), depth=3))
y_test = tf.squeeze(tf.one_hot(tf.cast(y_test, tf.int32), depth=3))

w = tf.Variable(tf.random.normal((4, 3)), dtype=tf.float32)
b = tf.Variable(tf.zeros(3), dtype=tf.float32)

epochs = 250
learning_rate = 0.001


for i in range(epochs):
    with tf.GradientTape() as tape:
        pre = tf.nn.softmax(x_train @ w + b)
        loss = -tf.reduce_sum(y_train * tf.math.log(pre))

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_train, axis=1), tf.argmax(pre, axis=1)), tf.float32))
    delta = tape.gradient(loss, [w, b])
    w.assign_sub(delta[0] * learning_rate), b.assign_sub(delta[1] * learning_rate)
    # 输出平均损失和准确率
    print("epoch：{}，loss：{}，accuracy：{}".format(i, loss / len(y_train), acc))

pre = tf.nn.softmax(x_test @ w + b)
loss = -tf.reduce_sum(y_test * tf.math.log(pre))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(pre, axis=1)), tf.float32))

print("test_loss：{}，test_accuracy：{}".format(loss / len(y_test), acc))
