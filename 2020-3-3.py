import math
x = input('请输入a, b, c的值（使用空格分开）：')
x = x.split(' ')
a, b, c = x
a, b, c = int(a), int(b), int(c)
delt = b * b - 4 * a * c
if delt >= 0:
    x1 = (-b + math.sqrt(delt)) / (2 * a)
    x2 = (-b - math.sqrt(delt)) / (2 * a)
    print("x1={},x2={}".format(x1, x2))
else:
    print('此方程无解')
