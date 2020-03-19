import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

plt.figure(figsize=(10, 10))
plt.rcParams['font.family'] = 'FZShuTi'

for i in range(16):
    num = np.random.randint(1, 60000)
    plt.subplot(4, 4, i + 1)
    plt.axis('off')
    plt.imshow(train_x[num], cmap='gray')
    plt.title('标签值：' + str(train_y[num]))

plt.tight_layout()
plt.show()
