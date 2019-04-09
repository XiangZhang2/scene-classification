# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from PIL import Image 
from PIL import ImageEnhance
import random
#import tensorflow as tf
import numpy as np
import random
import os 


# print('Process (%s) start...' % os.getpid())
# pid = os.fork()

# if pid == 0:
#     print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
# else:
#     print('I (%s) just created a child process (%s).' % (os.getpid(), pid))

image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_test_a_20180103/scene_test_a_images_20180103/'
image_name_1 = '0a02a8129d1298f2bed666bdb60e65a20abc67cf.jpg'
image_name_2 = '0a6442b84c7698d6beac357f1d254a3a9c71d78f.jpg'
image_name_3 = '0b5f3c75619bb2dc747b459e4409fa6fb1fd0537.jpg'

# image_list = []
# image_1 = Image.open(image_dir + image_name_1)
# image_1 = np.asarray(image_1)
# image_list.append(image_1)
# image_2 = Image.open(image_dir + image_name_2)
# image_2 = np.asarray(image_2)
# image_list.append(image_2)
# image_3 = Image.open(image_dir + image_name_3)
# image_3 = np.asarray(image_3)
# image_list.append(image_3)

# image_list = [a.astype(np.float32) for a in image_list]
# image_list = np.array(image_list)
# print image_list
# image_list /= 2
# print "*******"
# print image_list

#image_1 = plt.imread('miemie.jpg')
def a(image):
    image = np.asarray(image)
    image.flags.writeable = True
    return image

image_1 = Image.open(image_dir + image_name_1)
image_1 = a(image_1)
print image_1.flags.writeable
# plt.imshow(image_1)
# plt.show()
# miemie = image_1[500:700, 670:870]
# plt.imshow(miemie)
# plt.show()

# image_2 = plt.imread('woman.jpg')
# plt.imshow(image_2)
# plt.show()
# image_2[350:550, 400:600] = miemie
# plt.imshow(image_2)
# plt.show()
















