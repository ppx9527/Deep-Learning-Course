##文件说明
### style-transfer.py
这是用TensorFlow实现的，参考了TensorFlow的官方教程，但算法有很多可以优化的地方，总变分损失的优化不好，产生了较多的高频残差，
核心原理是基于Gatys的论文[A Neural Algorithm of Artistic Style]。<br>
整个模型在训练20轮数时，损失基本就不在变化，产生的图像风格基本已经接近风格图片。

### keras.py
这是Keras的版本其大致实现和TensorFlow的版本差别不大