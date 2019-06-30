import cv2
import numpy as np
from skimage.filters import sobel, gabor
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def watershed(image):
    if isinstance(image, list):
        image_matrix = list()
        for i,img in enumerate(image):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img, markers)
            img[markers == -1] = [255, 0, 0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_matrix.append(gray)

            if i == 0:
                plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
                plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(gray), plt.title('Watershed Filtered Image')
                plt.xticks([]), plt.yticks([])
                plt.show()

        return image_matrix
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(gray), plt.title('Watershed Filtered Image')
        plt.xticks([]), plt.yticks([])
        plt.show()
        return gray


def sobel_filter(image):
    if isinstance(image, list):
        image_matrix = list()
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s = sobel(img)
            image_matrix.append(s)

        return image_matrix
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        s = sobel(image)
        return s


def adaptive_histogram(image, clipLimit, tile_grid):
    if isinstance(image, list):
        image_matrix = list()
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile_grid, tile_grid))
            cl1 = clahe.apply(img)
            image_matrix.append(cl1)
        return image_matrix
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tile_grid, tile_grid))
        cl1 = clahe.apply(image)
        return cl1


def gabor_filter(image, frequency):
    if isinstance(image, list):
        image_matrix = list()
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filt_real, filt_imag = gabor(img, frequency=frequency)
            image_matrix.append(filt_real)
        return image_matrix
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filt_real, filt_imag = gabor(image, frequency=frequency)
        return filt_real


def adaptive_gaussian_thresholding(image, blur_factor, maxValue, blockSize, const):
    if isinstance(image, list):
        image_matrix = list()
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_cl1 = cv2.medianBlur(img, blur_factor)
            thresholded = cv2.adaptiveThreshold(blur_cl1, maxValue=maxValue, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=blockSize, C=const)
            image_matrix.append(thresholded)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_cl1 = cv2.medianBlur(image, blur_factor)
        thresholded = cv2.adaptiveThreshold(blur_cl1, maxValue=maxValue, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            thresholdType=cv2.THRESH_BINARY_INV, blockSize=blockSize, C=const)
        return thresholded
