# coding=utf-8
import os
import zipfile
#import matplotlib.pyplot as plt
from PIL import Image
from functools import reduce
import tensorflow as tf
import numpy as np
import json
import random
from keras.utils.np_utils import to_categorical

from data_augmentation import color_jitter, random_crop, get_random_eraser


def get_class(class_file):
    #tested
    with open(class_file) as f:
        lines = f.readlines()
        lines = [c.strip() for c in lines]
        class_list = [c.split(' ')[1] for c in lines]
    return class_list


def get_label(class_file, image_file, output_path):
    #tested
    class_list = get_class(class_file)
    with open(output_path + 'train_label.txt', 'w') as f:
        for class_name in class_list:
            name_list = os.listdir(image_file + class_name + '/')
            for name in name_list:
                f.write(class_name + '/' + name + ' ' + str(class_list.index(class_name)) + '\n')

# image = Image.open(image_file + 'Tomato_Tomv/59f9754c-2d49-4a74-ba11-4f9a017c8348___PSU_CG 2200.JPG')
# image.show()

def get_num(class_file, image_file, output_path):
    #tested
    class_list = get_class(class_file)
    with open(output_path + 'train_image_num.txt', 'w') as f:
        for class_name in class_list:
            name_list = os.listdir(image_file + class_name + '/')
            f.write(class_name + ' ' + str(len(name_list)) + '\n')


def unzip(zip_file, output_path):
    try:
        zip = zipfile.ZipFile(zip_file, 'r')
        zip.extractall(output_path)
        zip.close()
        print('unzip succeed!')
    except:
        print('unzip failed!')


def z_score(image):
    image = np.asarray(image)
    image = image / 255.
    mean = [0.4960301824223457, 0.47806493084428053, 0.44767167301470545]
    var = [0.084966025569294362, 0.082005493489533315, 0.088877477602068156]
    
    image[:,:,0] = (image[:,:,0] - mean[0]) / (var[0] ** 0.5)
    image[:,:,1] = (image[:,:,1] - mean[1]) / (var[1] ** 0.5)
    image[:,:,2] = (image[:,:,2] - mean[2]) / (var[2] ** 0.5)

    return image


def get_random_data(annotation_list, image_size, image_dir, is_training):
    
    image = Image.open(image_dir + annotation_list[1])

    if is_training:
        #color jitter
        image = color_jitter(image)

        #random flip left and right
        if np.random.uniform() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #random crop and resize
        image = random_crop(image, image_size)
        
        #random erasing
        image = np.asarray(image)
        eraser = get_random_eraser()
        image = eraser(image)

        image = z_score(image)

    else:
        ###for densetnet###
        # image = image.astype(np.float64)
        # image = image[:,:, ::-1]  #把RGB改成BGR了
        # image[:, :, 0] -= 103.939
        # image[:, :, 1] -= 116.779
        # image[:, :, 2] -= 123.68

        #####for inception_res-net v2
        input_shape = (image_size, image_size)
        image = image.resize(input_shape)
        #image = np.array(image)
        image = z_score(image)

    label = np.array(annotation_list[0])
    label = to_categorical(label, num_classes=80)  #modify there if classes is not 80

    return image, label


def get_json(annotation_file):
    # annotation=[[label, image_name],...] num=32739
    with open(annotation_file) as json_file:
        label_file = json.load(json_file)

        annotation_list = []

        for i in range(len(label_file)):
            annotation = [int(label_file[i]["label_id"]), label_file[i]["image_id"]]
            annotation_list.append(annotation)

    return annotation_list

# l = get_json('/Users/xiang/Downloads/dataset/AIC_disease/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json')
# print(l[1])
# print(len(l))

def label_shuffle(annotation_list):
    #类别平衡策略，将每个类别的图片数量扩充到和最大图片数量一致
    image_list_dic = {}
    image_num = []
    classes = 80

    for i in range(classes):
        image_list_dic[str(i)] = []
        for j in range(len(annotation_list)):
            if annotation_list[j][0] == i:
                image_list_dic[str(i)].append(annotation_list[j][1])
        image_num.append(len(image_list_dic[str(i)]))

    max_index = image_num.index(max(image_num))  #label_id=32  image_num=862
    min_index = image_num.index(min(image_num))  #label_id=55  image_num=168
    num_max = image_num[max_index]

    for n in range(classes):
        multiple = num_max // image_num[n]
        mod = num_max % image_num[n]
        copy = image_list_dic[str(n)]
        
        if multiple > 1:
            for m in range(multiple-1):
                image_list_dic[str(n)] = image_list_dic[str(n)] + copy
        
        image_list_dic[str(n)] += random.sample(copy, mod)

    annotation_list_with_shuffle = []
    for k in range(classes):
        for z in range(len(image_list_dic[str(k)])):
            annotation = [int(k), image_list_dic[str(k)][z]]
            annotation_list_with_shuffle.append(annotation)
    #print len(annotation_list_with_shuffle) # 68960
    np.random.shuffle(annotation_list_with_shuffle)

    return annotation_list_with_shuffle


def data_generator(annotation_list, batch_size, image_size, image_dir, is_training):
    '''data generator for fit_generator'''
    n = len(annotation_list)
    i = 0
    while True:
        image_data = []
        label_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_list)
            image, label = get_random_data(annotation_list[i], image_size, image_dir, is_training)
            #data augmentation following

            image_data.append(image)
            label_data.append(label)
            i = (i+1) % n
        image_data = np.array(image_data) #大概变成了(batch,416,416,3)
        label_data = np.array(label_data)
        yield (image_data, label_data)
        #yield表示函数生成器，每次产生的是一个batch的数据，但后面的batch_size是干嘛的？


def data_generator_wrapper(annotation_list, batch_size, image_size, image_dir, is_training=True):
    n = len(annotation_list)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_list, batch_size, image_size, image_dir, is_training)
























