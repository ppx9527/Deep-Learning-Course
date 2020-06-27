import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255, x_test / 255

model = tf.keras.Sequential([
    # 卷积后的窗口长度为：(图片长宽 - 内核大小) / (1 + 步长) + 1
    # 单个图像：image(32, 32, 3) + padding(2) => image(34, 34, 3) -convolution- kernel(3 * 3 * 3) + bias(1) => image(32, 32)
    # filters = 32时, stack(image(32, 32) * 32, axis=2) => (32, 32, 32)
    # input(n, 32, 32, 3) + filters(32, kernel(3 * 3 * 3) => output(n, 32, 32, 32)
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, input_shape=(32, 32, 3), padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_normal()
    ),

    # 为防止过拟合，每次有25%的权值不改变
    tf.keras.layers.Dropout(rate=0.25),

    # 添加池化层(基本操作与卷积一样，差别只在最后输出上)，进行降采样，使用最大池化。形状为(2, 2)的池化窗口，默认获得形状为(2, 2)的步幅
    # image(32, 32, 32) -maxPoll- pool(2 * 2) => image(16, 16, 32) => output(n, 16, 16, 32)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # image(16, 16, 32) + padding(2) => image(18, 18, 32) -convolution- kernel(3 * 3 * 32) + bias(1) => image(16, 16)
    # filters = 64时, stack(image(16, 16) * 64, axis=2) => (16, 16, 64)
    # input(n, 16, 16, 32) + filters(64, kernel(3 * 3 * 32) => output(n, 16, 16, 64)
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.he_uniform()
    ),

    # image(16, 16, 64) -maxPoll- pool(2 * 2) => image(8, 8, 64) => output(n, 8, 8, 64)
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # input(n, 8, 8, 64) => output(n, 8, 8, 128)
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, activation=tf.nn.relu, padding='same',
        kernel_initializer=tf.keras.initializers.he_normal()
    ),

    # input(n, 8, 8, 128) => output(n, 4, 4, 128)
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 添加平坦层：input(n, 4, 4, 128) -reshape(n, -1)- output(n, 2048)
    tf.keras.layers.Flatten(),

    # 全连接层：input(n, 2048) => output(n, 512) => output(n, 64) => output(n, 10)
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation=tf.nn.relu),

    # 输出层，经过softMax激活产生10分类的结果
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.summary()

train_epochs = 100
batch_size = 50

model.compile(
    optimizer=tf.optimizers.Adadelta(0.01),
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=train_epochs,
    validation_split=0.2
)

model.evaluate(x_test, y_test, verbose=2)
