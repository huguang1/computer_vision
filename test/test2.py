import cv2
import matplotlib.pyplot as plt
import numpy as np


def bilateral():
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        img = cv2.imread(i)
        blur = cv2.bilateralFilter(img, 15, 75, 75)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(blur[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def GaussianBlur():
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        img = cv2.imread(i)
        blur = cv2.GaussianBlur(img, (17, 17), 0)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(blur[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def blur():
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        img = cv2.imread(i)
        blur = cv2.blur(img, (9, 9))
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(blur[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def medianBlur():
    for i in ['20.jpg', '21.jpg', '15.jpg']:
        img = cv2.imread(i)
        medianBlur = cv2.medianBlur(img, 13)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(medianBlur[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def la(image):
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
    # kernel = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 0]], np.float32)
    kernel = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]], np.float32)
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst


def lapulasi():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        img = cv2.imread(i)
        medianBlur = la(img)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(medianBlur[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


if __name__ == '__main__':
    lapulasi()


