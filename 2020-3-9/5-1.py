import numpy as np
np.random.seed(618)
a = np.random.rand(1000)
s = int(input('请输入一个1-100之间的整数:'))
print('序号  索引值  随机数')
j = 1
for i in range(0, 1000):
    if i % s == 0:
        print("{}\t{}\t{}".format(j, i, a[i]))
        j += 1
