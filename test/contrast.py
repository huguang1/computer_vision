import cv2
import matplotlib.pyplot as plt
import numpy as np


def alpha_beta():
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        new_image = np.zeros(image.shape, image.dtype)
        alpha = 1.0  # Simple contrast control
        beta = 0  # Simple brightness control

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
            print(y)

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(new_image[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def convert_abs():
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    for i in ['10.jpg', '11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 10  # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(adjusted[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def gramma_vert():
    # https: // blog.csdn.net / bryant_meng / article / details / 111362422
    for i in ['10.jpg', '11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lookUpTable = np.empty((1, 256), np.uint8)
        for gamma in [0.6, 0.67, 0.7, 0.8, 0.9, 1]:
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            res = cv2.LUT(image, lookUpTable)
            plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.axis('off')
            plt.title('image')
            plt.imshow(image[:, :, [0, 1, 2]])
            plt.subplot(122)
            plt.axis('off')
            plt.title('filtered')
            plt.imshow(res[:, :, [0, 1, 2]])
            plt.tight_layout(True)
            plt.show()


def use_hist():
    # https://pythonmana.com/2022/04/202204100054590069.html
    for i in ['1.jpg', '7.jpg', '8.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (b, g, r) = cv2.split(image)
        # Equalize all three channels
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # Final merger
        result = cv2.merge((bH, gH, rH))
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(result[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def increase_contrast():
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    for i in ['10.jpg', '11.jpg', '12.jpg', '13.jpg']:
        img = cv2.imread(i, 1)
        # converting to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))
        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # Stacking the original image with the enhanced image
        result = np.hstack((img, enhanced_img))
        plt.figure(figsize=(10, 10))
        plt.imshow(result[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def add_weight():
    for i in ['10.jpg', '11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = 1.5
        beta = 20
        new_image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(new_image[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def contrast_brightness():
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    for i in ['10.jpg', '11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        contrast = 1.5
        brightness = 70
        brightness += int(round(255 * (1 - contrast) / 2))
        print(brightness)
        result = cv2.addWeighted(image, contrast, image, 0, brightness)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(result[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


def contrast_hsv():
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    for i in ['10.jpg', '11.jpg', '12.jpg', '13.jpg']:
        image = cv2.imread(i)
        imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imghsv[:, :, 2] = [[max(pixel - 25, 0) if pixel > 190 else min(pixel + 25, 255) for pixel in row] for row in
                           imghsv[:, :, 2]]
        result = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.axis('off')
        plt.title('image')
        plt.imshow(image[:, :, [0, 1, 2]])
        plt.subplot(122)
        plt.axis('off')
        plt.title('filtered')
        plt.imshow(result[:, :, [0, 1, 2]])
        plt.tight_layout(True)
        plt.show()


if __name__ == '__main__':
    contrast_brightness()






