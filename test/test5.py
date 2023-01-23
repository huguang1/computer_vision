import numpy as np


def gaussian_kernel(kernel_size, sigma):
    start = -1
    end = 1
    mean_x = 0
    mean_y = 0
    X = np.linspace(start, end, kernel_size)
    Y = np.linspace(start, end, kernel_size)
    x, y = np.meshgrid(X, Y)
    gaussian = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    return gaussian


sigma = 0.7
kernel = gaussian_kernel(3, sigma)

print(kernel)





