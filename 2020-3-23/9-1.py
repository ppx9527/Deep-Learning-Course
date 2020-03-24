import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x1 = tf.constant([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                  106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
x2 = tf.constant([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
y = tf.constant([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
                 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])


X = tf.stack((tf.ones(len(x1)), x1, tf.cast(x2, dtype=tf.float32)), axis=1)
Y = tf.reshape(y, (-1, 1))

# (X’X)-1 X'Y
# (3, 16) * (16 * 3) = (3 * 3) * (3 * 16) = (3 * 16) * (16 * 1) = (3 * 1）
W = tf.linalg.inv(tf.transpose(X) @ X) @ tf.transpose(X) @ Y
W = tf.squeeze(W)

print("请输入房屋面积和房间数，预测房屋销售价格")
while True:
    x1_test = float(input("商品房面积："))
    if x1_test < 20 or x1_test > 500:
        print('面积应是20-500之间的实数')
    else:
        break
while True:
    x2_test = int(input("房间数："))
    if x2_test < 1 or x2_test > 10:
        print('房间数应是1-10之间的整数')
    else:
        break

y_pre = W[1] * x1_test + W[2] * x2_test + W[0]

print("预测价格：", round(y_pre.numpy(), 2), "万元")
