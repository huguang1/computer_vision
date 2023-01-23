# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy


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


def CLAHE(I):
    lab = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    O = cv2.merge([l, a, b])
    O = cv2.cvtColor(O, cv2.COLOR_LAB2BGR)
    return O


def shapen_USM_C(I):
    row, col, dem = I.shape
    sigma = 3
    kernel_size = (13, 13)
    C = CLAHE(I)
    L = cv2.GaussianBlur(I, kernel_size, sigma)
    L = L.astype(numpy.int32)
    I = I.astype(numpy.int32)
    H = I - L
    H[H > 255] = 255
    H[H < 0] = 0
    lab_H = cv2.cvtColor(H.astype(numpy.uint8), cv2.COLOR_BGR2LAB)
    l_H = cv2.split(lab_H)[0]
    threshold = 1
    O = I.copy()
    for i in range(row):
        for j in range(col):
            for k in range(dem):
                percent = l_H[i, j] / 100
                diff = C[i, j, k] - I[i, j, k]
                delta = percent * diff
                if abs(delta) >= threshold:
                    O[i, j, k] += delta
    O[O > 255] = 255
    O[O < 0] = 0
    O = O.astype(numpy.uint8)
    return O


def shapen_USM(img):
    blur = GaussianFilter(img)
    blur = blur.astype(numpy.int32)
    img = img.astype(numpy.int32)
    change_img = img - blur
    a = 1.5
    out_img = img + a*change_img
    out_img[out_img > 255] = 255
    out_img[out_img < 0] = 0
    out_img = out_img.astype(numpy.uint8)
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





USM()


# for i in ['20.jpg', '21.jpg', '15.jpg']:
#     img = cv2.imread(i)
#     unsharp_image = GaussianFilter(img)
#     plt.figure(figsize=(10, 10))
#     plt.subplot(121)
#     plt.axis('off')
#     plt.title('image')
#     plt.imshow(img[:, :, [2, 1, 0]])
#     plt.subplot(122)
#     plt.axis('off')
#     plt.title('filtered')
#     plt.imshow(unsharp_image[:, :, [2, 1, 0]])
#     plt.tight_layout(True)
#     plt.show()








