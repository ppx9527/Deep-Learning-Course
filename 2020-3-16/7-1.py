from matplotlib import pyplot as plt
from PIL import Image


img = Image.open('lena.tiff')
img_r, img_g, img_b = img.split()

plt.figure(figsize=(10, 10))
plt.rcParams['font.family'] = 'FZShuTi'

plt.subplot(2, 2, 1)
plt.title('R-缩放', fontsize='14')
plt.axis('off')
plt.imshow(img_r.resize((50, 50)), cmap='gray')

plt.subplot(2, 2, 2)
plt.title('G-镜像+翻转', fontsize='14')
plt.imshow(img_g.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90), cmap='gray')

plt.subplot(2, 2, 3)
plt.title('B-裁剪', fontsize='14')
plt.axis('off')
plt.imshow(img_b.crop((0, 0, 300, 300)), cmap='gray')

plt.subplot(2, 2, 4)
plt.title('RGB', fontsize='14')
plt.axis('off')
img = Image.merge('RGB', [img_r, img_g, img_b])
img.save('lena.png')
plt.imshow(img)

plt.suptitle('图像基本操作', color='b', fontsize='20')
plt.show()
