import time

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import threading
import copy


# this funciton split image to r, g, b
def split_image(img):
    row, col, plane = img.shape
    r = np.zeros((row, col), np.uint8)
    r[:, :] = img[:, :, 0]
    g = np.zeros((row, col), np.uint8)
    g[:, :] = img[:, :, 1]
    b = np.zeros((row, col), np.uint8)
    b[:, :] = img[:, :, 2]
    return r, g, b


# this function merge r, g, b to a image
def merge_image(r, g, b):
    row, col = r.shape
    img = np.zeros((row, col, 3), np.uint8)
    img[:, :, 0] = r[:, :]
    img[:, :, 1] = g[:, :]
    img[:, :, 2] = b[:, :]
    return img


# This function will perform the convolution operation
def convolution(k, data, img_new):
    len_k = len(k)
    n, m = data.shape
    for i in range(n - len_k):
        line = []
        for j in range(m - len_k):
            a = data[i:i + len_k, j:j + len_k]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)


# Divide the image into three channels and use multiple threads to calculate
def filter_kernel(image, kernel):
    b, g, r = split_image(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=convolution, args=(kernel, r, r_c,))
    t2 = threading.Thread(target=convolution, args=(kernel, g, g_c,))
    t3 = threading.Thread(target=convolution, args=(kernel, b, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = merge_image(np.clip(np.array(r_c).astype(int), 0, 255), np.clip(np.array(g_c).astype(int), 0, 255),
                      np.clip(np.array(b_c).astype(int), 0, 255))
    return img


def rgb_gray(img):
    h, w, c = img.shape
    gray = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    gray_img = gray.reshape(h, w).astype(np.uint8)
    return gray_img


# first
def mean_kernel_blur():
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        image = cv2.imread(i)
        kernel = np.ones((9, 9), np.float32) / 81
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# second
def gaussian_kernel_blur():  # 高斯滤波
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        image = cv2.imread(i)
        sigma = 1
        kernel = gaussian_kernel(17, sigma)
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


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


# third 双边滤波
def bilateralFilter(img, radius, sigmaColor, sigmaSpace):
    B, G, R = split_image(img)
    B_tran, G_tran, R_tran = split_image(img)
    img_height = len(B)
    img_width = len(B[0])
    # 计算灰度值模板系数表
    color_coeff = -0.5 / (sigmaColor * sigmaColor)
    weight_color = []  # 存放颜色差值的平方
    for i in range(256):
        weight_color.append(np.exp(i * i * color_coeff))
    # 计算空间模板
    space_coeff = -0.5 / (sigmaSpace * sigmaSpace)
    weight_space = []  # 存放模板系数
    weight_space_row = []  # 存放模板 x轴 位置
    weight_space_col = []  # 存放模板 y轴 位置
    maxk = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r_square = i * i + j * j
            weight_space.append(np.exp(r_square * space_coeff))
            weight_space_row.append(i)
            weight_space_col.append(j)
            maxk = maxk + 1
    # 进行滤波
    for row in range(img_height):
        print(row)
        for col in range(img_width):
            value = 0
            weight = 0
            for i in range(maxk):
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width:
                    val = 0
                else:
                    val = B[m][n]
                w = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val - B[row][col])])
                value = value + val * w
                weight = weight + w
            B_tran[row][col] = np.uint8(value / weight)
    # 绿色通道
    for row in range(img_height):
        for col in range(img_width):
            value = 0
            weight = 0
            for i in range(maxk):
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width:
                    val = 0
                else:
                    val = G[m][n]
                w = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val - G[row][col])])
                value = value + val * w
                weight = weight + w
            G_tran[row][col] = np.uint8(value / weight)
    # 红色通道
    for row in range(img_height):
        for col in range(img_width):
            value = 0
            weight = 0
            for i in range(maxk):
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width:
                    val = 0
                else:
                    val = R[m][n]
                w = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val - R[row][col])])
                value = value + val * w
                weight = weight + w
            R_tran[row][col] = np.uint8(value / weight)
    return merge_image(B_tran, G_tran, R_tran)


def bilateral():
    for i in ['9.jpg', '20.jpg', '21.jpg', '15.jpg']:
        img = cv2.imread(i)
        bilateral_img = bilateralFilter(img, 15, 30, 80)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(bilateral_img[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


# fourth 中值滤波
def MedianFilter(img, K_size=13):
    # 中值滤波
    h, w, c = img.shape
    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float)
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad + y, pad + x, ci] = np.median(tmp[y:y + K_size, x:x + K_size, ci])
    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out


def median():
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        image = cv2.imread(i)
        img = MedianFilter(image)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


# fifth 高斯锐化
def gaussion_sharpenness():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        # 1.图片锐化后图片变得更加清晰
        KSIZE = 11
        ALPHA = 2
        sigma = 0.4
        kernel = -ALPHA * gaussian_kernel(KSIZE, sigma)
        kernel[KSIZE // 2, KSIZE // 2] += 1 + ALPHA
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# sixth 拉普拉斯锐化
def laplacian(img, K_size=3):
    H, W = img.shape
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()
    K = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = (-1) * np.sum(K * (tmp[y: y + K_size, x: x + K_size])) + tmp[pad + y, pad + x]
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * (tmp[y: y + K_size, x: x + K_size]))
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out


def laplacian_sharpen():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        r, g, b = split_image(image)
        out_r = laplacian(r, K_size=3)
        out_g = laplacian(g, K_size=3)
        out_b = laplacian(b, K_size=3)
        out = merge_image(out_r, out_g, out_b)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(out[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


# seventh
def GaussianFilter(img):
    h, w, c = img.shape
    K_size = 13
    sigma = 1.3
    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float)
    # 定义滤波核
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (sigma * np.sqrt(2 * np.pi))
    K /= K.sum()
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad + y, pad + x, ci] = np.sum(K * tmp[y:y + K_size, x:x + K_size, ci])
    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out


def shapen_USM(img):
    blur = GaussianFilter(img)
    blur = blur.astype(np.int32)
    w, h, c = blur.shape
    img = img.astype(np.int32)[:w, :h, :]
    change_img = img - blur
    a = 1.5
    out_img = img + a*change_img
    out_img[out_img > 255] = 255
    out_img[out_img < 0] = 0
    out_img = out_img.astype(np.uint8)
    return out_img


def USM():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        img = cv2.imread(i)
        unsharp_image = shapen_USM(img)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(unsharp_image[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


if __name__ == "__main__":
    a = time.time()
    USM()
    print(time.time() - a)

