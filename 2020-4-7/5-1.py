import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# 生成数据
x_data = np.linspace(-1, 1, 500)
y_data = 3.1234 * x_data + 2.98 + np.random.randn(*x_data.shape) * 0.4


def model(w, x, b):
    return tf.multiply(w, x) + b


def loss(x, y, w, b):
    err = y - model(w, x, b)
    return tf.reduce_mean(tf.square(err))


def gard(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
        return tape.gradient(loss_, [w, b])


w = tf.Variable(np.random.rand(), tf.float32)
b = tf.Variable(0.0)
training_epochs = 10
learning_rate = 0.005
step = 0
display_step = 20

for epoch in range(training_epochs):
    for xs, ys in zip(x_data, y_data):
        loss_ = loss(xs, ys, w, b)
        delta_w, delta_b = gard(xs, ys, w, b)
        w.assign_sub(delta_w * learning_rate)
        b.assign_sub(delta_b * learning_rate)

        step += 1
        if step % display_step == 0:
            print("Train Epoch{}, Step:{}, loss:{:0.6f}".format(epoch + 1, step, loss_))

print(w.numpy(), b.numpy())
print(model(w, 5.79, b))
