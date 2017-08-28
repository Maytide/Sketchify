# http://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html
"""
Convert 256 x 256 images into binary images used for training data, using
helper function from image_utils.py
"""

import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from image_utils import *
from collect_anime_pics import detect_pics

RESIZE_FOLDER = 'resized'
MANGIFY_FOLDER = 'mangify'

check_types=('*.png', '*.jpg', '*.jpeg', '*.gif', '*.tiff', '*.bmp')

def mangify_folder(openpath='resized_256_256', savepath='mangify_256_256'):
    current_imgs = set(detect_pics(folder=savepath))
    img_files = detect_pics(folder=openpath)

    for i, img_file in enumerate(img_files[:]):
        img = load_image(img_file, openpath='')
        img = filter_contour(img)
        img_file = img_file.split('\\')[1]

        if os.path.join(savepath, img_file.replace('.png', '.jpg')) not in current_imgs:
            fullpath = save_image(img, img_file, savepath=savepath, ext='jpg')
            current_imgs.add(fullpath)
            print(i, 'Successfully mangified:', img_file)
        else:
            print('Image:', img_file, 'already exists.')
            continue
        # break

if __name__ == '__main__':
    mangify_folder(openpath='sketchify/testing/resized_3',
                   savepath='sketchify/testing/mangify_3')
