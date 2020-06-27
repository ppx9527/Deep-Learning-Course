"""
风格迁移：生成图像与基本图像具有相同的内容，但是具有不同图片的“风格”（通常是艺术性的）
这是通过优化损失函数来实现的它有3个损失：“风格损失”、“内容损失”，以及“总变化损失”

-总变分损失使得组合图像的像素，使其具有视觉连贯性。
-风格的损失是风格迁移的核心，使用多个深卷积神经网络生产多个输出再生成Gram矩阵。用来捕捉不同空间的颜色/纹理信息
-内容损失是基特征之间的L2距离图像（从深层提取）和组合图像的特征，使生成的图像与原始图像足够接近。
参考文献
-[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from matplotlib import pyplot as plt
import PIL.Image
import numpy as np


# 加载图片，并将其最大尺寸限制为 512 像素
def load_img(path_to_img, max_dim=512):
    # 加载图片并进行归一化
    img = PIL.Image.open(path_to_img)
    img = np.array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 选择最长的维度，获得缩放的比例
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    # 对图像进行缩放，并添加维度
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.expand_dims(img, axis=0)
    return img


# 加载风格图片和需要转换的图片
content_image = load_img('image/Sunlit-Mountains.jpg')
style_image = load_img('image/seated-nude.jpg')

# 选择生产特征图的卷积层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

content_layers = ['block5_conv2']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# 返回由选择的层组成的模型
def vgg_layers(layer_names):
    # 加载模型，加载已经在imageNet数据上预训练的VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # 获取需要的卷积层，作为输出添加到模型中
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # 每个卷积层都有会输出一个特征图
    model = tf.keras.Model([vgg.input], outputs)
    return model


#  通过在每个位置计算feature(特征图)向量的外积，并在所有位置对该外积进行平均,可以计算出包含此信息的 Gram 矩阵
def gram_matrix(feature):
    # 获得卷积后的图像
    x = feature[0]

    # 将通道数提到最前面
    x = tf.transpose(x, perm=[2, 0, 1])

    # 计算Gram = A * At
    x = tf.reshape(x, [x.shape[0], -1])  # x = [ch, w * h]
    result = x @ tf.transpose(x)
    result = tf.expand_dims(result, axis=0)

    # G / w * h
    input_shape = tf.shape(feature)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


# 生成特征图
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    @tf.function
    def call(self, inputs):
        # 期望浮点输入 [0,1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # 进行前向计算返回风格和内容的特征图
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        # 对每个特征图进行计算得到Gram矩阵
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # 将内容特征图和卷积层的名字存入字典
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        # 将Gram矩阵和卷积层的名字存入字典
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# 获取目标风格和内容特征图
extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(tf.constant(style_image))['style']  # 风格图片的Gram矩阵
content_targets = extractor(tf.constant(content_image))['content']  # 原图片的特征图

# 定义超参数
image = tf.Variable(content_image)  # 生成风格的图像
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)  # 使用Adam优化器
style_weight = 0.01  # 风格权重，权重越大风格生成图片风格改变越明显
content_weight = 1000  # 内容图片权重，权重越大与原图片越接近
total_variation_weight = 300  # 总变分损失的权重


# 将图片归一化
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# 返回高频残差
def high_pass_x_y(image):
    # 计算相邻像素值的差。
    x_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    y_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    return x_var, y_var


# 相关的正则化损失是这些值的平方和
def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)


# -损失计算，使用均方差损失
# -风格损失：风格图片的Gram矩阵 - 内容图片的Gram矩阵，让损失最小，来使其风格接近
# -内容损失：生成的内容图片的特征图 - 内容图片本来的特征图，让损失最小，来使其特征接近，保持生成的图片和原有图片内容大致不变
def style_content_loss(outputs):
    # 得到内容图片的Gram矩阵和特征图
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    # 风格损失
    style_loss = tf.add_n(
        [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    # 特征图损失
    content_loss = tf.add_n(
        [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    # 返回总损失
    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * total_variation_loss(image)  # 加上产生的高频残差

    # 应用梯度
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# 开始生成
epochs = 15
steps_per_epoch = 100

for n in range(epochs):
    for m in range(steps_per_epoch):
        train_step(image)

    # 输出每一轮的损失
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_loss(image)
    print("epoch：{}，loss：{}".format(n, loss))

plt.imshow(image.read_value()[0])
plt.show()

# 保存图片
file_name = 'image/stylized-image.jpg'


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


tensor_to_image(image).save(file_name)
