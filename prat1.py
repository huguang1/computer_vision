import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading
import math


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
    r, g, b = split_image(image)
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


# 1.This function is to realize the sketch of the picture
def dodgeV2(image, mask):
    return image * 256 / (255 - mask)


def rgb_to_sketch(img_rgb):
    img_gray = rgb_gray(img_rgb)
    img_gray_inv = 255 - img_gray
    img_blur = GaussianFilter(img_gray_inv)
    img_blend = dodgeV2(img_gray, img_blur)
    return img_blend


# -*- coding: utf-8 -*-
def GaussianFilter(img):
    h, w = img.shape
    K_size = 21
    sigma = 3
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float)
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (sigma * np.sqrt(2 * np.pi))
    K /= K.sum()
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            out[pad + y, pad + x] = np.sum(K * tmp[y:y + K_size, x:x + K_size])
    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out


def rgb_gray(img):
    h, w, c = img.shape
    gray = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    gray_img = gray.reshape(h, w).astype(np.uint8)
    return gray_img


def sketch():
    for i in ['33.jpg', '34.jpg', '35.jpg']:
        img_rgb = cv2.imread(i)
        dst = rgb_to_sketch(img_rgb)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img_rgb[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(dst, cmap='gray')
        plt.tight_layout(True)
        plt.show()


# 2. This filter is to achieve the effect of hip hop mirror
def enlarge_effect(img, cx, cy, radius):
    h, w, n = img.shape
    r = int(radius / 2.0)
    new_img = img.copy()
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy
            distance = tx * tx + ty * ty
            if distance < radius * radius:
                x = int(int(tx / 2.0) * (math.sqrt(distance) / r) + cx)
                y = int(int(ty / 2.0) * (math.sqrt(distance) / r) + cy)
                if x < w and y < h:
                    new_img[j, i, 0] = img[y, x, 0]
                    new_img[j, i, 1] = img[y, x, 1]
                    new_img[j, i, 2] = img[y, x, 2]
    return new_img


def enlarge():
    for i in ['33.jpg', '34.jpg', '35.jpg']:
        image = cv2.imread(i)
        cx, cy, radius = 500, 500, 300
        img = enlarge_effect(image, cx, cy, radius)
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


# 3. The effect of this filter is to achieve the effect of relief
def relief(img):
    h, w, c = img.shape
    img_gray = rgb_gray(img)
    reliefImg = np.zeros((h, w, 1), np.uint8)

    for i in range(h - 1):
        for j in range(w - 1):
            edge = int(img_gray[i, j]) - int(img_gray[i, j + 1])   # 得到边缘
            edge += int(img_gray[i, j]) - int(img_gray[i + 1, j])  # 得到边缘
            val = edge + 150  # 产生立体感
            if val > 255:
                val = 255
            if val < 0:
                val = 0
            reliefImg[i, j] = val
    return reliefImg


def relief_picture():
    for i in ['33.jpg', '34.jpg', '35.jpg']:
        image = cv2.imread(i)
        img = relief(image)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(img, cmap='gray')
        plt.tight_layout(True)
        plt.show()


# 4. This filter is for light effect
def light_effect():
    for i in ['35.jpg', '33.jpg', '34.jpg', ]:
        img = cv2.imread(i)
        rows, cols = img.shape[:2]
        centerX = rows / 5
        centerY = cols / 2
        radius = min(centerX, centerY)
        strength = 200
        dst = np.zeros((rows, cols, 3), dtype="uint8")
        for i in range(rows):
            for j in range(cols):
                distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
                B = img[i, j][0]
                G = img[i, j][1]
                R = img[i, j][2]
                if (distance < radius * radius):
                    result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                    G = img[i, j][1] + result
                    R = img[i, j][2] + result
                    B = img[i, j][2] - result * 0.5
                    B = min(255, max(0, B))
                    G = min(255, max(0, G))
                    R = min(255, max(0, R))
                    dst[i, j] = np.uint8((B, G, R))
                else:
                    dst[i, j] = np.uint8((B, G, R))
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(dst[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


# 5. This function implements is to convert the picture into a cartoon style
def threshold(img):
    h, w = img.shape
    thresz_img = np.zeros((h, w), np.uint8)
    k = 5
    for i in range(h-k):
        for j in range(w - 1-k):
            mean = np.mean(img[i:i+k, j: j+k])
            val = int(img[i, j])
            if val > mean/1.04:
                val = 255
            else:
                val = 0
            thresz_img[i, j] = val
    return thresz_img


def MedianFilter(img, K_size=5):
    # 中值滤波
    h, w = img.shape
    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float)
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            out[pad + y, pad + x] = np.median(tmp[y:y + K_size, x:x + K_size])
    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out


def cortoon_effect(img):
    h, w, c = img.shape
    img_color = img
    kernel = np.ones((7, 7), np.float32) / 49
    img_color = filter_kernel(img_color, kernel)
    img_gray = rgb_gray(img)
    img_blur = MedianFilter(img_gray, 7)
    img_edge = threshold(img_blur)
    h_c, w_c, c = img_color.shape
    edge = np.empty([h, w, c], dtype=int)
    edge[:, :, 0] = img_edge
    edge[:, :, 1] = img_edge
    edge[:, :, 2] = img_edge
    new_img = img_color & edge[:h_c, :w_c, :]
    return new_img


def cortoon():
    for i in ['24.jpg', '25.jpg', '32.jpg']:
        image = cv2.imread(i)
        img = cortoon_effect(image)
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


if __name__ == "__main__":
    cortoon()

