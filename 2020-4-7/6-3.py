import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import numpy as np

buston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_test, y_test) = buston_housing.load_data()
x_train, x_test = tf.cast(scale(x_train), tf.float32), tf.cast(scale(x_test), tf.float32)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)


def loss(x, y, w, b):
    pre = x @ w + b
    loss_ = tf.reduce_mean(tf.square(y - pre))
    return loss_


def gard(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
        return tape.gradient(loss_, [w, b])


w = tf.Variable(tf.random.normal([13, 1], mean=0, stddev=1.), dtype=tf.float32)
b = tf.Variable(tf.zeros(1), dtype=tf.float32)

train_epochs = 100
learning_rate = 0.001
batch_size = 10
optimizer = tf.keras.optimizers.SGD(learning_rate)

# 小批量梯度下降
for epoch in range(train_epochs):
    for step in range(int(len(x_train) / batch_size)):
        xs = x_train[step * batch_size:(step + 1) * batch_size]
        ys = y_train[step * batch_size:(step + 1) * batch_size]

        gards = gard(xs, ys, w, b)
        optimizer.apply_gradients(zip(gards, [w, b]))

    train_loss = loss(x_train, y_train, w, b)  # 当前轮次总的损失
    print("epoch：{:3d}，train_loss：{:.4f}，".format(epoch, train_loss))

print(w.numpy(), '\n', b.numpy())
house_id = np.random.randint(0, len(x_test))
pre = tf.reshape(x_test[house_id], (1, -1)) @ w + b
print("第{}条数据，预测值：{}，实际值：{}".format(house_id, pre.numpy(), y_test[house_id]))

plt.figure()
plt.subplot(121)
plt.plot(y_train, marker="o", label="true")
plt.plot(x_train @ w + b, marker=".", label="pre")

plt.subplot(122)
plt.plot(y_test, marker="o", label="true")
plt.plot(x_test @ w + b, marker=".", label="pre")
plt.show()
