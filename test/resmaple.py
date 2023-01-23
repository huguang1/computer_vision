import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading


# 1: Image Bilinear Interpolation Algorithm
def filter_kernel(src, new_size, img_new):
    # 目标图像宽高
    dst_h, dst_w = new_size
    # 源图像宽高
    src_h, src_w = src.shape
    # 如果图像大小一致，直接复制返回即可
    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    # 计算缩放比例
    scale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_h
    # 遍历目标图像
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
    # return dst
    # 对通道进行循环
    # for n in range(3):
    # 对 height 循环
    for dst_y in range(dst_h):
        # 对 width 循环
        line = []
        for dst_x in range(dst_w):
            # 目标在源上的坐标
            src_x = dst_x * scale_x
            src_y = dst_y * scale_y
            # 计算在源图上 4 个近邻点的位置
            # i,j
            i = int(np.floor(src_x))
            j = int(np.floor(src_y))
            u = src_x-i
            v = src_y-j
            if j == src_h-1:
                j = src_h-2
            if i == src_w-1:
                i = src_h-2
            kernel = np.array([[(1-u)*(1-v), u*(1-v)],
                               [(1-u)*v, u*v]])
            a = np.array([[src[j, i], src[j+1, i]],
                          [src[j, i+1], src[j+1, i+1]]])
            line.append(np.sum(np.multiply(kernel, a)))
        img_new.append(line)


# Divide the image into three channels and use multiple threads to calculate
def resize_bilinear(image, new_size):
    r, g, b = cv2.split(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=filter_kernel, args=(r, new_size, r_c,))
    t2 = threading.Thread(target=filter_kernel, args=(g, new_size, g_c,))
    t3 = threading.Thread(target=filter_kernel, args=(b, new_size, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = cv2.merge([np.array(r_c).astype(int), np.array(g_c).astype(int), np.array(b_c).astype(int)])
    return img


# 2: average
def filter_kernel_average(src, new_size, img_new):
    # 目标图像宽高
    dst_h, dst_w = new_size
    # 源图像宽高
    src_h, src_w = src.shape
    # 如果图像大小一致，直接复制返回即可
    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    # 计算缩放比例
    scale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_h
    # 遍历目标图像
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
    # return dst
    # 对通道进行循环
    # for n in range(3):
    # 对 height 循环
    kernel = np.array([[0.25, 0.25],
                       [0.25, 0.25]])
    for dst_y in range(dst_h):
        # 对 width 循环
        line = []
        for dst_x in range(dst_w):
            # 目标在源上的坐标
            src_x = dst_x * scale_x
            src_y = dst_y * scale_y
            # 计算在源图上 4 个近邻点的位置
            # i,j
            i = int(np.floor(src_x))
            j = int(np.floor(src_y))
            if j >= src_h-1:
                j = src_h-2
            if i >= src_w-1:
                i = src_h-2
            a = np.array([[src[j, i], src[j+1, i]],
                          [src[j, i+1], src[j+1, i+1]]])
            line.append(np.sum(np.multiply(kernel, a)))
        img_new.append(line)


# Divide the image into three channels and use multiple threads to calculate
def resize_average(image, new_size):
    r, g, b = cv2.split(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=filter_kernel_average, args=(r, new_size, r_c,))
    t2 = threading.Thread(target=filter_kernel_average, args=(g, new_size, g_c,))
    t3 = threading.Thread(target=filter_kernel_average, args=(b, new_size, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = cv2.merge([np.array(r_c).astype(int), np.array(g_c).astype(int), np.array(b_c).astype(int)])
    return img


# 3: nearest points
def filter_nearest(src, new_size, img_new):
    dst_y, dst_x = new_size
    src_y, src_x = src.shape
    # 生成一个黑色的目标图像
    dst = np.zeros((dst_y, dst_x), dtype=np.uint8)
    scale_x = float(src_x) / dst_x
    scale_y = float(src_y) / dst_y
    # 渲染像素点的值
    # 注意 y 是高度，x 是宽度
    for y in range(dst_y-1):
        line = []
        for x in range(dst_x-1):
            # 计算新坐标 (x,y) 坐标在源图中是哪个值
            i = round(x * scale_x)
            j = round(y * scale_y)
            if j >= src_y-1:
                j = src_y-2
            if i >= src_x-1:
                i = src_x-2
            line.append(src[j, i])
        img_new.append(line)


def resize_nearest(image, new_size):
    r, g, b = cv2.split(image)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=filter_nearest, args=(r, new_size, r_c,))
    t2 = threading.Thread(target=filter_nearest, args=(g, new_size, g_c,))
    t3 = threading.Thread(target=filter_nearest, args=(b, new_size, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = cv2.merge([np.array(r_c).astype(int), np.array(g_c).astype(int), np.array(b_c).astype(int)])
    return img


# 4.像素邻域的双三次插值
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
    print(img.shape)
    pad_image = np.zeros((h + 4, w + 4))
    pad_image[2:h + 2, 2:w + 2] = img
    return pad_image


def bicubic(img, sacle, new_img, a=-0.5):
    h, w = img.shape
    img = padding(img)  # 将图片先放大一点
    nh = int(h * sacle)
    nw = int(w * sacle)
    for i in range(nw):
        line = []
        for j in range(nh):
            px = i / sacle + 2
            py = j / sacle + 2
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


def resize_bicubic(img, sacle):
    r, g, b = cv2.split(img)
    r_c, g_c, b_c = [], [], []
    t1 = threading.Thread(target=bicubic, args=(r, sacle, r_c,))
    t2 = threading.Thread(target=bicubic, args=(g, sacle, g_c,))
    t3 = threading.Thread(target=bicubic, args=(b, sacle, b_c,))
    t1.start()
    t2.start()
    t3.start()
    for p in [t1, t2, t3]:
        p.join()
    img = cv2.merge([np.array(r_c).astype(int).T, np.array(g_c).astype(int).T, np.array(b_c).astype(int).T])
    return img


#
def minify():
    insert = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    for i in ['24.jpg', '25.jpg', '26.jpg', '27.jpg', '28.jpg']:
        for c in insert:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=c)
            # 1.图片锐化后图片变得更加清晰
            plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.axis('off')
            plt.title('image')
            plt.imshow(image[:, :, [0, 1, 2]])
            plt.subplot(122)
            plt.axis('off')
            plt.title('filtered')
            plt.imshow(resized_img[:, :, [0, 1, 2]])
            plt.tight_layout(True)
            plt.show()


def magnify():
    insert = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    for i in ['24.jpg', '25.jpg', '26.jpg', '27.jpg', '28.jpg']:
        for c in insert:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            resized_img = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=c)
            resized_height, resized_width, channel = resized_img.shape
            # 1.图片锐化后图片变得更加清晰
            plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.axis('off')
            plt.title('image')
            plt.imshow(image[int(height/8):int(height/4), int(width/8):int(width/4), [0, 1, 2]])
            plt.subplot(122)
            plt.axis('off')
            plt.title('filtered')
            plt.imshow(resized_img[int(resized_height/8):int(resized_height/4), int(resized_width/8):int(resized_width/4), [0, 1, 2]])
            plt.tight_layout(True)
            plt.show()


if __name__ == '__main__':
    ab = time.time()
    sacle = (500, 1000)
    src = cv2.imread('24.jpg')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    dst = resize_bicubic(src, sacle)
    print(time.time() - ab)
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.axis('off')
    plt.title('image')
    plt.imshow(src)
    plt.subplot(122)
    plt.axis('off')
    plt.title('filtered')
    plt.imshow(dst)
    plt.tight_layout(True)
    plt.show()











