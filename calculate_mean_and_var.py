from PIL import Image
import numpy as np
from tqdm import tqdm
import os

image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_train_20170904/scene_train_images_20170904/'
#image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'

image_list = os.listdir(image_dir)

try:
    index = image_list.index('.DS_Store')
    image_list = image_list[:index] + image_list[index+1:]
except:
    image_list = image_list

r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0

for i in range(len(image_list)):
    image = Image.open(image_dir + image_list[i])
    image = np.asarray(image)
    image = image / 255.
    total = total + image.shape[0] * image.shape[1]

    r += image[:, :, 0].sum()
    g += image[:, :, 1].sum()
    b += image[:, :, 2].sum()

    r_2 += (image[:, :, 0] ** 2).sum()
    g_2 += (image[:, :, 1] ** 2).sum()
    b_2 += (image[:, :, 2] ** 2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

print('Mean is %s' % ([r_mean, g_mean, b_mean]))
print('Var is %s' % ([r_var, g_var, b_var]))


