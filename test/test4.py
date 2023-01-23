import numpy as np
import matplotlib.pyplot as plt


x1 = 0.32
y1 = np.round(np.sin(0.32), 6)
x2 = 0.34
y2 = np.round(np.sin(0.34), 6)
x3 = 0.36
y3 = np.round(np.sin(0.36), 6)


# Lagrange插值函数
def lagrange_f(x, x1, x2, x3, y1, y2, y3):
    l1 = y1 * (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
    l2 = y2 * (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
    l3 = y3 * (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))
    return l1 + l2 + l3


def f(x):
    return np.sin(x)


x = np.linspace(0, np.pi, 1000)
y = f(x)

x_ = 0.3367
result = np.round(lagrange_f(0.3367, x1, x2, x3, y1, y2, y3), 6)
print(result)
new_y = lagrange_f(x, x1, x2, x3, y1, y2, y3)
# 原图和拟合图
plt.figure(figsize=(10, 8), dpi=80)
plt.plot(x, y, 'b')
plt.plot(x, new_y, 'g')
plt.plot(x1, y1, 'r*')
plt.plot(x2, y2, 'r*')
plt.plot(x3, y3, 'r*')
plt.plot(x_, result, 'y*')
plt.show()
