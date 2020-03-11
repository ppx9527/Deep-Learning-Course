from matplotlib import pyplot as plt
import tensorflow as tf

buston_housing = tf.keras.datasets.boston_housing
(x, y), (_, _) = buston_housing.load_data(test_split=0)
titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
          "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT", "MEDV"]

plt.rcParams['font.family'] = 'FZShuTi'
plt.figure(figsize=(10, 10))


def scatter_generator(index):
    plt.scatter(x[:, index], y)
    plt.xlabel(titles[index])
    plt.ylabel('Price($1000\'s)')
    plt.title(str(index + 1) + '.' + titles[index] + ' - Price')


for i in range(13):
    plt.subplot(4, 4, i + 1)
    scatter_generator(i)

plt.tight_layout()
plt.suptitle('各个属性与房价的关系', x=0.5, y=1.02, fontsize=18)
plt.show()

plt.figure()
print('''
    1--CRIM\n2--ZN\n3--INDUS\n4--CHAS\n5--NOX\n6--RM\n7AGE
    \n8--DIS\n9--RAD\n10-TAX\n11-PTRATIO\n12-B\n13-LSTAT
''')
s = int(input('请输入属性：')) - 1
scatter_generator(s)
plt.show()
