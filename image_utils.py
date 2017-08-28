"""
Additional utility functions used for image manipulation/processing.
"""


import sys, os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Load image RGB
def load_image(img, openpath=''):
    imgpath = os.path.join(openpath, img)

    img_rgb = cv2.imread(imgpath)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    return img_rgb


# Load image greyscale
def load_image_grayscale(img, openpath=''):
    imgpath = os.path.join(openpath, img)

    img_rgb = cv2.imread(imgpath)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    return img_gray


# Display image with matplotlib
def show_image(img, grayscale=True):
    if not grayscale:
        imgplot = plt.imshow(img)
    else:
        imgplot = plt.imshow(img, cmap='gray')
    plt.show()

	
# http://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html
# Image to sketch method used to generate sketch ground truth images.
def image_to_sketch(img, openpath=''):

    def dodge_v2(image, mask):
        return cv2.divide(image, 255-mask, scale=256)

    def burn_v2(image, mask):
        return 255 - cv2.divide(255-image, 255-mask, scale=256)

    if isinstance(img, str):
        img_gray = load_image_grayscale(img, openpath=openpath)
    else:
        print('Image stored as format:', type(img))
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as ex:
            print(ex)
            raise ValueError('Unknown image type.')

    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)

    img_blend = dodge_v2(img_gray, img_blur)

    return img_blend

#######################################################

# From Deepcolor repository
# Converts image to binary greyscale using adaptive thresholding.
def filter_contour(img_rgb):
    img_contour = cv2.adaptiveThreshold(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, blockSize=9, C=2)

    img_denoise = cv2.medianBlur(img_contour, 3)

    return img_denoise

#######################################################

# Save image, replacing extension to jpg
def save_image(img, img_name, savepath='', ext=None):
    if ext is not None:
        ext = ext.replace('.', '').replace('jpeg', 'jpg')
        assert ext in ('jpg', 'png')
        img_name = img_name.split('.')
        img_name = img_name[:-1]
        img_name.append(ext)
        # print(img_name, img_name[:-1])
        img_name = '.'.join(img_name)
        # print(img_name)

    fullpath = os.path.join(savepath, img_name)
    cv2.imwrite(fullpath, img)
    print('Saved image:', img_name, 'in folder:', savepath)
    return fullpath

# Save image, keeping extension
def save_image_v2(img, img_name, savepath):
    fullpath = os.path.join(savepath, img_name)
    cv2.imwrite(fullpath, img)
    print('Saved image:', img_name, 'in folder:', savepath)
    return fullpath