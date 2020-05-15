import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from datetime import datetime

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

model = tf.keras.models.Sequential([
    # input(n, 32, 32) --> output(n, 784)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # input(n, 784) * kernel(784, 64) = (n, 64) + bias(64) --> output(n, 64)
    tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer='normal'),
    # input(n, 64) * kernel(64, 32) = (n, 32) + bias(32) --> output(n, 32)
    tf.keras.layers.Dense(units=32, activation=tf.nn.relu, kernel_initializer='normal'),
    # input(n, 32) * kernel(32, 10) = (n, 10) + bias(10) --> output(n, 10)
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.summary()
model.compile(
    optimizer=tf.optimizers.Adam(0.001),
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

# 定义日志目录，启用TensorBoard
time = datetime.now().strftime("%Y%m%d-%H-%M-%S")
log_dir = os.path.join('..', 'logs', time)
os.mkdir(log_dir)
tensorBoard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    batch_size=30,
    epochs=5,
    callbacks=[tensorBoard_callback]
)

model.evaluate(x_test, y_test)
