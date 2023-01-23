import cv2
import time
import matplotlib.pyplot as plt


def resize_nearest(scale_list):
    for i in ['25.jpg', '37.jpg']:
        src = cv2.imread(i)
        dst_list = []
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)
            dst_list.append(dst)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        cv2.imwrite(r'near1.5{}.jpg'.format(i), dst_list[0])
        plt.subplot(223)
        plt.axis('off')
        plt.title('2 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        cv2.imwrite(r'near2{}.jpg'.format(i), dst_list[1])
        plt.subplot(224)
        plt.axis('off')
        plt.title('4 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        cv2.imwrite(r'near4{}.jpg'.format(i), dst_list[2])
        plt.tight_layout(True)
        plt.show()


def resize_nearest1(scale):
    for i in ['27.jpg', '37.jpg', '24.jpg', '25.jpg']:
        src = cv2.imread(i)

        # src1 = cv2.GaussianBlur(src, (3, 3), 0)
        height, width = src.shape[:2]  # 图片的高度和宽度
        dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)

        # dst1 = cv2.GaussianBlur(dst, (3, 3), 0)
        height, width = dst.shape[:2]  # 图片的高度和宽度
        dst2 = cv2.resize(dst, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)

        # dst3 = cv2.GaussianBlur(dst2, (3, 3), 0)
        height, width = dst2.shape[:2]  # 图片的高度和宽度
        dst4 = cv2.resize(dst2, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_NEAREST)

        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst[:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst2[:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst4[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def resize_bilinear(scale_list):
    for i in ['24.jpg', '27.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)
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
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        cv2.imwrite('bi1.5{}.jpg'.format(i), dst_list[0])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        cv2.imwrite('bi2{}.jpg'.format(i), dst_list[1])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        cv2.imwrite(r'bi4{}.jpg'.format(i), dst_list[2])
        plt.tight_layout(True)
        plt.show()


def resize_bilinear1(scale):
    for i in ['27.jpg', '37.jpg', '24.jpg', '25.jpg']:
        src = cv2.imread(i)

        # src1 = cv2.GaussianBlur(src, (3, 3), 0)
        height, width = src.shape[:2]  # 图片的高度和宽度
        dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)

        # dst1 = cv2.GaussianBlur(dst, (3, 3), 0)
        height, width = dst.shape[:2]  # 图片的高度和宽度
        dst2 = cv2.resize(dst, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)

        # dst3 = cv2.GaussianBlur(dst2, (5, 5), 0)
        height, width = dst2.shape[:2]  # 图片的高度和宽度
        dst4 = cv2.resize(dst2, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LINEAR)

        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst[:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst2[:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst4[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def resize_cubic(scale_list):
    for i in ['25.jpg', '37.jpg']:
        src = cv2.imread(i)
        dst_list = []
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)
            dst_list.append(dst)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        cv2.imwrite(r'cu1.5{}.jpg'.format(i), dst_list[0])
        plt.subplot(223)
        plt.axis('off')
        plt.title('2 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        cv2.imwrite(r'cu2{}.jpg'.format(i), dst_list[1])
        plt.subplot(224)
        plt.axis('off')
        plt.title('4 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        cv2.imwrite(r'cu4{}.jpg'.format(i), dst_list[2])
        plt.tight_layout(True)
        plt.show()


def resize_cubic1(scale):
    for i in ['27.jpg', '37.jpg', '24.jpg', '25.jpg']:
        src = cv2.imread(i)

        # src1 = cv2.GaussianBlur(src, (3, 3), 0)
        height, width = src.shape[:2]  # 图片的高度和宽度
        dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)

        # dst1 = cv2.GaussianBlur(dst, (3, 3), 0)
        height, width = dst.shape[:2]  # 图片的高度和宽度
        dst2 = cv2.resize(dst, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)

        # dst3 = cv2.GaussianBlur(dst2, (3, 3), 0)
        height, width = dst2.shape[:2]  # 图片的高度和宽度
        dst4 = cv2.resize(dst2, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)

        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst[:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst2[:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst4[:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def resize_area(scale_list):
    for i in ['25.jpg', '37.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_AREA)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('0.5 image')
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.25 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('0.125 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def resize_area1(scale_list):
    for i in ['27.jpg', '37.jpg', '24.jpg', '25.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_AREA)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('image')
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def resize_lanczos4(scale_list):
    for i in ['27.jpg', '37.jpg', '24.jpg', '25.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LANCZOS4)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('image')
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


def resize_lanczos41(scale_list):
    for i in ['27.jpg', '37.jpg', '24.jpg', '25.jpg']:
        src = cv2.imread(i)
        dst_list = []
        a = time.time()
        for scale in scale_list:
            height, width = src.shape[:2]  # 图片的高度和宽度
            dst = cv2.resize(src, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_LANCZOS4)
            dst_list.append(dst)
        print(time.time() - a)
        plt.figure(figsize=(10, 9.8))
        plt.subplot(221)
        plt.axis('off')
        plt.title('image')
        plt.imshow(src[:, :, [2, 1, 0]])
        plt.subplot(222)
        plt.axis('off')
        plt.title('image')
        plt.imshow(dst_list[0][:, :, [2, 1, 0]])
        plt.subplot(223)
        plt.axis('off')
        plt.title('0.5 times')
        plt.imshow(dst_list[1][:, :, [2, 1, 0]])
        plt.subplot(224)
        plt.axis('off')
        plt.title('1.5 times')
        plt.imshow(dst_list[2][:, :, [2, 1, 0]])
        plt.tight_layout(True)
        plt.show()


# resize_nearest([0.5, 0.25, 0.125])
# resize_nearest([1.5, 2, 4])
resize_bilinear([0.5, 0.25, 0.125])
# resize_bilinear1(0.5)
# resize_cubic([1.5, 2, 4])
# resize_cubic1(0.5)
# resize_area([0.5, 0.25, 0.125])
# resize_lanczos4([0.25, 0.5, 2])






