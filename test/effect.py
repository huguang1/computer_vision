import cv2
import matplotlib.pyplot as plt
import numpy as np
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
    # a = np.array(r_c)
    # b = np.array(r_c).astype(int)
    # c = np.clip(b, 0, 255)
    img = merge_image(np.clip(np.array(r_c).astype(int), 0, 255), np.clip(np.array(g_c).astype(int), 0, 255), np.clip(np.array(b_c).astype(int), 0, 255))
    return img


# 1.This function can realize the characteristics of relief
def relief():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
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


# 2.?????????, ?????????????????????
def simple_contrast():
    # ?????????????????????????????????
    for i in ['11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.array([[0, 0, 0],
                           [0, 2, 0],
                           [0, 0, 0]])
        img = filter_kernel(image, kernel)
        # ??????????????????????????????????????????
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


# 3.gamma???????????????
def gamma_contrast():
    for i in ['11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(image)
        r_fI = r / 255.0
        gamma = 0.6
        r_out_image = (np.power(r_fI, gamma) * 255).astype(int)
        g_fI = g / 255.0
        g_out_image = (np.power(g_fI, gamma) * 255).astype(int)
        b_fI = b / 255.0
        b_out_image = (np.power(b_fI, gamma) * 255).astype(int)
        img = cv2.merge([r_out_image, g_out_image, b_out_image])
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


# 4.????????????????????????
def rule_contrast():
    # ?????????????????????????????????
    for i in ['11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, bi = cv2.split(image)

        r_max = np.max(r)
        r_min = np.min(r)
        # ?????????????????????????????????????????????
        # ??????a ??? b??????
        a = 255 / (r_max - r_min)
        b = 0 - a * r_min
        # ?????????????????????
        r_out_image = a * r + b
        # ?????????????????????
        r_out_image = r_out_image.astype(np.uint8)

        g_max = np.max(g)
        g_min = np.min(g)
        # ?????????????????????????????????????????????
        # ??????a ??? b??????
        a = 255 / (g_max - g_min)
        b = 0 - a * g_min
        # ?????????????????????
        g_out_image = a * g + b
        # ?????????????????????
        g_out_image = g_out_image.astype(np.uint8)

        bi_max = np.max(bi)
        bi_min = np.min(bi)
        # ?????????????????????????????????????????????
        # ??????a ??? b??????
        a = 255 / (bi_max - bi_min)
        b = 0 - a * bi_min
        # ?????????????????????
        bi_out_image = a * bi + b
        # ?????????????????????
        bi_out_image = bi_out_image.astype(np.uint8)
        img = cv2.merge([r_out_image, g_out_image, bi_out_image])
        # img = cv2.filter2D(image, -1, kernel)
        # ??????????????????????????????????????????
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


if __name__ == '__main__':
    rule_contrast()























