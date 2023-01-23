import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading


def split_image(img):
    row, col, plane = img.shape
    r = np.zeros((row, col), np.uint8)
    r[:, :] = img[:, :, 0]
    g = np.zeros((row, col), np.uint8)
    g[:, :] = img[:, :, 1]
    b = np.zeros((row, col), np.uint8)
    b[:, :] = img[:, :, 2]
    return r, g, b


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
    for i in range(n-len_k):
        line = []
        for j in range(m-len_k):
            a = data[i:i+len_k, j:j+len_k]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)


# Divide the image into three channels and use multiple threads to calculate
def filter_kernel(image, kernel):
    r, g, b = split_image(image)
    # r, g, b = cv2.split(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=convolution, args=(kernel, r, r_c,))
    t2 = threading.Thread(target=convolution, args=(kernel, g, g_c,))
    t3 = threading.Thread(target=convolution, args=(kernel, b, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    # img = cv2.merge([np.array(r_c).astype(int), np.array(g_c).astype(int), np.array(b_c).astype(int)])
    img = merge_image(np.array(r_c).astype(int), np.array(g_c).astype(int), np.array(b_c).astype(int))
    return img


# first
def mean_kernel_blur():
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.ones((5, 5), np.float32) / 25
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def cv_mean_kernel_blur():
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.ones((5, 5), np.float32) / 25
        blur3 = cv2.filter2D(image, -1, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(blur3[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def blur():  # 使用blur滤波器，这个用的应该就是均值模糊
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_mean = cv2.blur(image, (5, 5))
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img_mean[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# second
def gaussian_kernel_blur():  # 高斯滤波
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = gaussian_kernel(7)
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def gaussian_kernel(kernel_size):
    start = -1
    end = 1
    mean_x = 0
    mean_y = 0
    sigma = 1
    X = np.linspace(start, end, kernel_size)
    Y = np.linspace(start, end, kernel_size)
    x, y = np.meshgrid(X, Y)
    gaussian = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma ** 2))
    gaussian = gaussian/gaussian.sum()
    return gaussian


def gaussian_blur():  # 高斯滤波
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_Guassian = cv2.GaussianBlur(image, (9, 9), 0)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img_Guassian[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# third
def gaussion_sharpenness():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 1.图片锐化后图片变得更加清晰
        KSIZE = 11
        ALPHA = 2
        # 使用 <code>cv.getGaussianKernel</code>函数可以创建自己定义的高斯内核
        kernel = cv2.getGaussianKernel(KSIZE, 0)
        kernel = -ALPHA * kernel @ kernel.T
        kernel[KSIZE // 2, KSIZE // 2] += 1 + ALPHA
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def bilateral_filter():  # 双边滤波
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_bilater = cv2.bilateralFilter(image, 9, 75, 75)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img_bilater[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# -*- coding: UTF-8 -*-
def bilateralFilter(img, radius, sigmaColor, sigmaSpace):
    B, G, R = cv2.split(img)
    B_tran, G_tran, R_tran = cv2.split(img)
    img_height = len(B)
    img_width = len(B[0])
    # 计算灰度值模板系数表
    color_coeff = -0.5 / (sigmaColor * sigmaColor)
    weight_color = []       # 存放颜色差值的平方
    for i in range(256):
        weight_color.append(np.exp(i * i * color_coeff))
    # 计算空间模板
    space_coeff = -0.5 / (sigmaSpace * sigmaSpace)
    weight_space = []     # 存放模板系数
    weight_space_row = []  # 存放模板 x轴 位置
    weight_space_col = []  # 存放模板 y轴 位置
    maxk = 0
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            r_square = i*i + j*j
            r = np.sqrt(r_square)
            weight_space.append(np.exp(r_square * space_coeff))
            weight_space_row.append(i)
            weight_space_col.append(j)
            maxk = maxk + 1
    # 进行滤波
    for row in range(img_height):
        for col in range(img_width):
            value = 0
            weight = 0
            for i in range(maxk):
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width :
                    val = 0
                else :
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
            for i in range(maxk) :
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width :
                    val = 0
                else :
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
            for i in range(maxk) :
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width :
                    val = 0
                else:
                    val = R[m][n]
                w = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val - R[row][col])])
                value = value + val * w
                weight = weight + w
            R_tran[row][col] = np.uint8(value / weight)
    cv2.imshow("beauty_after", cv2.merge([B_tran, G_tran, R_tran]))
    cv2.imwrite("beauty_after.png", cv2.merge([B_tran, G_tran, R_tran]))


# img = cv2.imread("30.jpg")
# # cv2.imshow("original image", img)
#
# # bilateralFilter(img, 5, 45, 100)
# bilateralFilter(img, 3, 10, 30)
#
# img = cv2.imread("beauty_after.png")
# bilateralFilter(img, 3, 30, 80)

# cv2.waitKey(0)


def mediaxxn_blur():  # 中值滤波
    for i in ['15.jpg', '19.jpg', '20.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_median = cv2.medianBlur(image, 7)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img_median[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# forth
def simple_sharpe_kernel():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# fifth
def own_sharpe_kernel():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.array([[0, -0.1, -0.2, -0.1, 0],
                            [-0.1, -0.2, -0.3, -0.2, -0.1],
                            [-0.2, -0.3, 4.6, -0.3, -0.2],
                            [-0.1, -0.2, -0.3, -0.2, -0.1],
                            [0, -0.1, -0.2, -0.1, 0]])
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# sixth
def own_sharpe_kernel2():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
        img = filter_kernel(image, kernel)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


if __name__ == "__main__":
    image = cv2.imread("15.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = MedianFilter(image)
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.axis('off')
    plt.title('image')
    plt.imshow(image[:, :, [0, 1, 2]])
    plt.subplot(122)
    plt.axis('off')
    plt.title('filtered')
    plt.imshow(img[:, :, [0, 1, 2]])
    plt.tight_layout(True)
    plt.show()

