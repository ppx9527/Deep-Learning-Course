import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 划分数据和数据归一化并对标签进行独热编码
train_num = int(len(x_train) * 0.8)  # 训练集的数量
x_valid, y_valid = tf.cast(x_train[train_num:].reshape(-1, 784) / 255, tf.float32), y_train[train_num:]
x_train, y_train = tf.cast(x_train[:train_num].reshape(-1, 784) / 255, tf.float32), y_train[:train_num]
x_test, y_test = tf.cast(x_test.reshape(-1, 784) / 255, tf.float32), tf.one_hot(y_test, depth=10)
y_train, y_valid = tf.one_hot(y_train, depth=10), tf.one_hot(y_valid, depth=10)

# 超参数和变量
train_epochs = 15
learning_rate = 0.001
batch_size = 50
w = tf.Variable(tf.random.normal((784, 10)))
b = tf.Variable(tf.zeros(10))
optimizer = tf.keras.optimizers.Adam(learning_rate)

# 画图的列表
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []


def loss(x, y):
    pre = tf.nn.softmax(x @ w + b)
    loss_ = tf.keras.losses.categorical_crossentropy(y, pre)  # 交叉熵损失函数
    return tf.reduce_mean(loss_)


def accuracy(x, y):
    pre = tf.nn.softmax(x @ w + b)
    correct_pre = tf.equal(tf.argmax(pre, axis=1), tf.argmax(y, axis=1))  # 产生一个是否匹配的布尔型列表
    return tf.reduce_mean(tf.cast(correct_pre, tf.float32))


# 训练
for epoch in range(train_epochs):
    # 小批量梯度下降
    for step in range(int(train_num / batch_size)):
        xs = x_train[step * batch_size:(step + 1) * batch_size]
        ys = y_train[step * batch_size:(step + 1) * batch_size]

        with tf.GradientTape() as tape:
            loss_t = loss(xs, ys)

        delta = tape.gradient(loss_t, [w, b])
        optimizer.apply_gradients(zip(delta, [w, b]))

    loss_t, loss_v = loss(x_train, y_train), loss(x_valid, y_valid)
    acc_t, acc_v = accuracy(x_train, y_train), accuracy(x_valid, y_valid)
    train_loss.append(loss_t), valid_loss.append(loss_v)
    train_acc.append(acc_t), valid_acc.append(acc_v)
    print("epoch：{}，train_loss：{:.4f}，train_acc：{:.4f}，valid_loss：{:.4f}，valid_acc：{:.4f}"
          .format(epoch + 1, loss_t, acc_t, loss_v, acc_v))

acc_test = accuracy(x_test, y_test)
print("test accuracy：{:.4f}".format(acc_test))

plt.figure()
plt.subplot(311)
plt.title('loss')
plt.plot(train_loss, label='train')
plt.plot(valid_loss, label="valid")
plt.legend()

plt.subplot(312)
plt.title('accuracy')
plt.plot(train_acc, label='train')
plt.plot(valid_acc, label="valid")
plt.legend()

pre_t = tf.argmax(tf.nn.softmax(x_test @ w + b), axis=1).numpy()
y_test = tf.argmax(y_test, axis=1)
x_test = x_test * 255
num = np.random.randint(0, len(x_test))
plt.subplot(313)
plt.title('label:{}, pre:{}'.format(pre_t[num], y_test[num]))
plt.imshow(tf.reshape(x_test[num], (28, 28)), cmap='binary')
plt.tight_layout()
plt.show()
