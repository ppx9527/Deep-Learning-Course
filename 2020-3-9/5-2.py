import numpy as np

x = np.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y = np.array([162.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

t1 = t2 = 0
for i in range(0, len(x)):
    t1 += (x[i] - x.mean()) * (y[i] - y.mean())
    t2 += (x[i] - x.mean()) ** 2
w = t1 / t2
b = y.mean() - w * x.mean()
print("w:{}\nb:{}".format(w, b))
