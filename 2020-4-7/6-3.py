import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import scale
import numpy as np

buston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_test, y_test) = buston_housing.load_data()
x_train, x_test = tf.cast(scale(x_train), tf.float32), tf.cast(scale(x_test), tf.float32)


def model(w, x, b):
    return x @ w + b


def loss(x, y, w, b):
    err = y - model(w, x, b)
    return tf.reduce_mean(tf.square(err))


def gard(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
        return tape.gradient(loss_, [w, b])


w = tf.Variable(tf.random.normal([13, 1], mean=0, stddev=1.), dtype=tf.float32)
b = tf.Variable(tf.zeros(1), dtype=tf.float32)

train_epochs = 50
learning_rate = 0.001
batch_size = 10
optimizer = tf.keras.optimizers.SGD(learning_rate)

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
pre = model(w, x_test, b)[house_id]
print("第{}条数据，预测值：{:.4f}，实际值：{}".format(house_id, pre[0], y_test[house_id]))
