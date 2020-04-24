import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# 生成数据
x_data = np.linspace(0, 100, 500)
y_data = 3.1234 * x_data + 2.98 + np.random.randn(*x_data.shape) * 0.4


def gard(x, y, w, b):
    with tf.GradientTape() as tape:
        pre = tf.multiply(w, x) + b
        loss = tf.reduce_mean(tf.square(y - pre))
    return loss, tape.gradient(loss, [w, b])


# b的值设置为接近正确值的原因是由于学习率过小b的值改变很慢
# w的值在x = 30时就已经接近3.14了，这时损失很小，但随着x的增大损失也会变大（y - y' 变大），会使w变大。如果学习率过大，
# 就会导致w增长的过快，产生震荡导致梯度爆炸。所以学习率必须设置得很小
w = tf.Variable(np.random.rand(), tf.float32)
b = tf.Variable(3.0)
training_epochs = 10
learning_rate = 0.00001
step = 0
display_step = 20

for epoch in range(training_epochs):
    for xs, ys in zip(x_data, y_data):
        loss, (delta_w, delta_b) = gard(xs, ys, w, b)

        # w = w - n * delta,w的值被改变为梯度 * 学习率
        w.assign_sub(delta_w * learning_rate)
        b.assign_sub(delta_b * learning_rate)

        step += 1
        if step % display_step == 0:
            print("Train Epoch：{}, Step:{}, loss:{:0.6f}".format(epoch + 1, step, loss))
            print(w.numpy())

print(w.numpy(), b.numpy())
print('实际值：{}，预测值：{}'.format(3.14 * 5.79 + 2.98, tf.multiply(w, 5.97) + b))
