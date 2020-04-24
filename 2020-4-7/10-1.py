import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import numpy as np

buston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_test, y_test) = buston_housing.load_data()
x_train = tf.cast(tf.concat([scale(x_train), np.ones(len(x_train)).reshape(-1, 1)], axis=1), tf.float32)
x_test = tf.cast(tf.concat([scale(x_test), np.ones(len(x_test)).reshape(-1, 1)], axis=1), tf.float32)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

w = tf.Variable(tf.random.normal([14, 1], mean=0, stddev=1.), dtype=tf.float32)
train_epochs = 300
learning_rate = 0.01
mse_train, mse_test = [], []

# 批量梯度下降
for i in range(train_epochs):
    with tf.GradientTape() as tape:
        pre = x_train @ w
        loss = tf.reduce_mean(tf.square(y_train - pre))

        pre_t = x_test @ w
        loss_t = tf.reduce_mean(tf.square(y_test - pre_t))

    mse_train.append(loss)
    mse_test.append(loss_t)

    delta = tape.gradient(loss, w)
    w.assign_sub(delta * learning_rate)

pre_t = x_test @ w
plt.figure()
plt.subplot(131)
plt.plot(mse_test, label="test")
plt.plot(mse_train, label="train")
plt.legend()

plt.subplot(132)
plt.plot(y_train, marker="o", label="true")
plt.plot(x_train @ w, marker=".", label="pre")
plt.legend()

plt.subplot(133)
plt.plot(y_test, marker="o", label="true")
plt.plot(pre_t, marker=".", label="pre")
plt.legend()
plt.show()
