import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from matplotlib import pyplot as plt
import PIL.Image
import numpy as np

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# 选择最大激活的层
layer_names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


def read_image(file_name, max_dim=None):
    img = PIL.Image.open(file_name)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# 将图像进行切割移动
def random_roll(image, max_roll):
    # 产生随机的shift值
    shift = tf.random.uniform(shape=[2], minval=-max_roll, maxval=max_roll, dtype=tf.int32)
    shift_down, shift_right = shift[0], shift[1]
    # 按照随机的shift把图片的下面和上面部分进行交换，把左、右部分进行交换
    img_rolled = tf.roll(tf.roll(image, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled


# 损失是所选层中激活的总和。损耗在每一层均归一化，因此较大层的贡献不会超过较小层。在DeepDream中，通过梯度上升使这种损失最大化
def calc_loss(image, model):
    # 必须添加一个维度，batch_size，使图像变为(1, 150, 150, 3)
    img_batch = tf.expand_dims(image, axis=0)

    # 图像通过模型进行前向计算
    layer_activations = model(img_batch)

    # 如果batch为一，则计算两次损失
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []  # 计算每层的计算结果的均值
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


def show(img):
    plt.imshow(img)


# 返回卷积后的图片的梯度
class TiledGradients(tf.keras.models.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # 规定输入的图像的格式
    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32)
        )
    )
    # 定义call方法，来计算损失，应用梯度
    def __call__(self, img, tile_size=512):
        shift_down, shift_right, img_rolled = random_roll(img, tile_size)

        # 初始化梯度为0
        gradients = tf.zeros_like(img_rolled)
        print(img_rolled)

        # 产生分块坐标列表
        xs = range(0, img_rolled.shape[0], tile_size)
        ys = range(0, img_rolled.shape[1], tile_size)

        for x in xs:
            for y in ys:
                # 计算图块的梯度
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)

                    # 从图像中提取该图块
                    img_piece = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss = calc_loss(img_piece, self.model)

                # 更新梯度
                gradients = gradients + tape.gradient(loss, img_rolled)

        # 将进行移动的图块放回原来的位置
        gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

        # 归一化梯度
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        print(gradients)

        return xs


get_tiled_gradients = TiledGradients(dream_model)


def render_deep_dream(img, step_size=0.01, step_per_octave=100, octaves=range(1), octave_scale=1.3):
    # 对输入图像进行标准化
    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)

    for octave in octaves:
        # 进行图像缩放
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(step_per_octave):
            gradients = get_tiled_gradients(img)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                print("Octave：{}，Step：{}".format(octave, step))

    # img = tf.image.resize(img, base_shape)
    img = tf.cast(255 * (img + 1.0) / 2.0, tf.uint8)
    # img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
    return img


original_img = read_image('19.jpg')
# show(render_deep_dream(img=original_img))

a = get_tiled_gradients(img=original_img)
print(a)
plt.show()


