import random
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading


# this funciton split image to r, g, b
def split_image(img):
    row, col, plane = img.shape
    r = np.zeros((row, col), np.uint8)
    g = np.zeros((row, col), np.uint8)
    b = np.zeros((row, col), np.uint8)
    b[:, :] = img[:, :, 0]
    g[:, :] = img[:, :, 1]
    r[:, :] = img[:, :, 2]
    return r, g, b


# this function merge r, g, b to a image
def merge_image(r, g, b):
    row, col = r.shape
    img = np.zeros((row, col, 3), np.uint8)
    img[:, :, 0] = r[:, :]
    img[:, :, 1] = g[:, :]
    img[:, :, 2] = b[:, :]
    return img


def change_shape(img, new_dst):
    new_dst[:, :, 0], new_dst[:, :, 1], new_dst[:, :, 2] = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return new_dst


# Divide the image into three channels and use multiple threads to calculate
def filter_kernel_bicubic(image, scale, function):
    r, g, b = split_image(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=function, args=(r, scale, r_c,))
    t2 = threading.Thread(target=function, args=(g, scale, g_c,))
    t3 = threading.Thread(target=function, args=(b, scale, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = merge_image(np.clip(np.array(r_c).astype(int).T, 0, 255), np.clip(np.array(g_c).astype(int).T, 0, 255),
                      np.clip(np.array(b_c).astype(int).T, 0, 255))
    return img


def filter_kernel(image, scale, function):
    r, g, b = split_image(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=function, args=(r, scale, r_c,))
    t2 = threading.Thread(target=function, args=(g, scale, g_c,))
    t3 = threading.Thread(target=function, args=(b, scale, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = merge_image(np.clip(np.array(r_c).astype(int), 0, 255), np.clip(np.array(g_c).astype(int), 0, 255),
                      np.clip(np.array(b_c).astype(int), 0, 255))
    return img


# 1: nearest points
def resize_nearest(scale_list):
    for i in ['25.jpg', '37.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            dst = filter_kernel(src, scale, filter_biliner)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[0][:, :, [0, 1, 2]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('2 times')
        plt.imshow(dst_list[1][:, :, [0, 1, 2]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('4 times')
        plt.imshow(dst_list[2][:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def filter_nearest(src, scale, img_new):
    src_y, src_x = src.shape
    dst_y, dst_x = round(src_y * scale), round(src_x * scale)
    for y in range(dst_y - 1):
        line = []
        for x in range(dst_x - 1):
            # 计算新坐标 (x,y) 坐标在源图中是哪个值
            i = round(x / scale)
            j = round(y / scale)
            if j >= src_y - 1:
                j = src_y - 2
            if i >= src_x - 1:
                i = src_x - 2
            line.append(src[j, i])
        img_new.append(line)


# 2: gaussian
def resize_gaussian(scale_list):
    for i in ['25.jpg', '37.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            print(scale)
            dst = filter_kernel(src, scale, filter_kernel_gaussian)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[0][:, :, [0, 1, 2]])
        cv2.imwrite(r'gau1.5{}.jpg'.format(i), dst_list[0][:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('2 times')
        plt.imshow(dst_list[1][:, :, [0, 1, 2]])
        cv2.imwrite(r'gau2{}.jpg'.format(i), dst_list[1][:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('4 times')
        plt.imshow(dst_list[2][:, :, [0, 1, 2]])
        cv2.imwrite(r'gau4{}.jpg'.format(i), dst_list[2][:, :, [2, 1, 0]])
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


def filter_kernel_gaussian(src, scale, img_new):
    src_h, src_w = src.shape
    dst_h, dst_w = round(src_h * scale), round(src_w * scale)
    sigma = 0.7
    kernel = gaussian_kernel(3, sigma)
    for dst_y in range(dst_h):
        line = []
        for dst_x in range(dst_w):
            src_x = dst_x / scale
            src_y = dst_y / scale
            i = int(np.floor(src_x))
            j = int(np.floor(src_y))
            if j >= src_h - 1:
                j = src_h - 2
            if i >= src_w - 1:
                i = src_w - 2
            a = np.array([[src[j - 1, i - 1], src[j, i - 1], src[j + 1, i - 1]],
                          [src[j - 1, i], src[j, i], src[j + 1, i]],
                          [src[j - 1, i + 1], src[j, i + 1], src[j + 1, i + 1]]
                          ])
            line.append(np.sum(np.multiply(kernel, a)))
        img_new.append(line)


# 3.像素邻域的双三次插值
def base_function(x, a=-0.5):
    # describe the base function sin(x)/x
    Wx = 0
    if np.abs(x) <= 1:
        Wx = (a + 2) * (np.abs(x) ** 3) - (a + 3) * x ** 2 + 1
    elif 1 <= np.abs(x) <= 2:
        Wx = a * (np.abs(x) ** 3) - 5 * a * (np.abs(x) ** 2) + 8 * a * np.abs(x) - 4 * a
    return Wx


def padding(img):
    h, w = img.shape
    pad_image = np.zeros((h + 4, w + 4))
    pad_image[2:h + 2, 2:w + 2] = img
    return pad_image


def bicubic(img, scale, new_img, a=-0.5):
    h, w = img.shape
    img = padding(img)  # 将图片先放大一点
    nh = int(h * scale)
    nw = int(w * scale)
    for i in range(nw):
        line = []
        for j in range(nh):
            px = i / scale + 2
            py = j / scale + 2
            px_int = int(px)
            py_int = int(py)
            u = px - px_int
            v = py - py_int
            A = np.matrix([[base_function(u + 1, a)], [base_function(u, a)], [base_function(u - 1, a)],
                           [base_function(u - 2, a)]])
            C = np.matrix(
                [base_function(v + 1, a), base_function(v, a), base_function(v - 1, a), base_function(v - 2, a)])
            B = np.matrix([[img[py_int - 1, px_int - 1], img[py_int - 1, px_int],
                            img[py_int - 1, px_int + 1], img[py_int - 1, px_int + 2]],
                           [img[py_int, px_int - 1], img[py_int, px_int], img[py_int, px_int + 1],
                            img[py_int, px_int + 2]],
                           [img[py_int + 1, px_int - 1], img[py_int + 1, px_int],
                            img[py_int + 1, px_int + 1], img[py_int + 1, px_int + 2]],
                           [img[py_int + 2, px_int - 1], img[py_int + 2, px_int],
                            img[py_int + 2, px_int + 1], img[py_int + 2, px_int + 2]]])
            line.append([np.dot(np.dot(C, B), A)][0][0, 0])
        new_img.append(line)


def resize_bicubic(scale_list):
    for i in ['25.jpg', '37.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            dst = filter_kernel(src, scale, bicubic)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[0][:, :, [0, 1, 2]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('2 times')
        plt.imshow(dst_list[1][:, :, [0, 1, 2]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('4 times')
        cv2.imwrite('new1.jpg', dst_list[2][:, :, [2, 1, 0]])
        plt.imshow(dst_list[2][:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


# 4: Image Bilinear Interpolation Algorithm
def resize_bilinear(scale_list):
    for i in ['27.jpg', '24.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            dst = filter_kernel(src, scale, filter_biliner)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst_list[0][:, :, [0, 1, 2]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst_list[1][:, :, [0, 1, 2]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst_list[2][:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def filter_biliner(src, scale, img_new):
    src_h, src_w = src.shape
    dst_h, dst_w = int(src_h * scale), int(src_w * scale)
    for dst_y in range(dst_h):
        line = []
        for dst_x in range(dst_w):
            src_x = dst_x / scale
            src_y = dst_y / scale
            i = int(np.floor(src_x))
            j = int(np.floor(src_y))
            u = src_x - i
            v = src_y - j
            if j == src_h - 1:
                j = src_h - 2
            if i == src_w - 1:
                i = src_w - 2
            kernel = np.array([[(1 - u) * (1 - v), u * (1 - v)],
                               [(1 - u) * v, u * v]])
            a = np.array([[src[j, i], src[j + 1, i]],
                          [src[j, i + 1], src[j + 1, i + 1]]])
            line.append(np.sum(np.multiply(kernel, a)))
        img_new.append(line)


# 5. average
def resize_average(scale_list):
    for i in ['27.jpg', '24.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            dst = filter_kernel(src, scale, filter_kernel_average)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst_list[0][:, :, [0, 1, 2]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst_list[1][:, :, [0, 1, 2]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst_list[2][:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def filter_kernel_average(src, scale, img_new):
    src_h, src_w = src.shape
    dst_h, dst_w = round(src_h * scale), round(src_w * scale)
    if scale > 0.5:
        kernel = [[0.25, 0.25],
                  [0.25, 0.25]]
    elif scale > 0.25:
        kernel = [[0.11, 0.11, 0.11],
                  [0.11, 0.11, 0.11],
                  [0.11, 0.11, 0.11]]
    else:
        kernel = [[0.0625, 0.0625, 0.0625, 0.0625],
                  [0.0625, 0.0625, 0.0625, 0.0625],
                  [0.0625, 0.0625, 0.0625, 0.0625],
                  [0.0625, 0.0625, 0.0625, 0.0625]]

    for dst_y in range(dst_h):
        line = []
        for dst_x in range(dst_w):
            src_x = dst_x / scale
            src_y = dst_y / scale
            i = round(np.floor(src_x))
            j = round(np.floor(src_y))
            if j >= src_h - 2:
                j = src_h - 3
            if i >= src_w - 2:
                i = src_w - 3
            if scale > 0.5:
                a = np.array([[src[j, i], src[j + 1, i]],
                              [src[j, i + 1], src[j + 1, i + 1]]])
            elif scale > 0.25:
                a = np.array([[src[j - 1, i - 1], src[j, i - 1], src[j + 1, i - 1]],
                              [src[j - 1, i], src[j, i], src[j + 1, i]],
                              [src[j - 1, i + 1], src[j, i + 1], src[j + 1, i + 1]]
                              ])
            else:
                a = np.array([[src[j - 1, i - 1], src[j, i - 1], src[j + 1, i - 1], src[j + 2, i - 1]],
                              [src[j - 1, i], src[j, i], src[j + 1, i], src[j + 2, i]],
                              [src[j - 1, i + 1], src[j, i + 1], src[j + 1, i + 1], src[j + 2, i + 1]],
                              [src[j - 1, i + 2], src[j, i + 2], src[j + 1, i + 2], src[j + 2, i + 2]]
                              ])
            line.append(np.sum(np.multiply(kernel, a)))
        img_new.append(line)


if __name__ == '__main__':
    scale = [0.5, 0.25, 0.125]
    resize_average(scale)
