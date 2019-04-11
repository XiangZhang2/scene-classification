# coding=utf-8
import os
import multiprocessing
import zipfile
#import matplotlib.pyplot as plt
from PIL import Image
from functools import reduce
import tensorflow as tf
import numpy as np
import json
import random

from data_augmentation import color_jitter, random_crop, load_data
from data_augmentation import aug_images_single, z_score, color_jitter


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
                f.write(class_name + '/' + name + ' ' + 
                    str(class_list.index(class_name)) + '\n')


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


def get_json(annotation_file):
    # annotation=[[label, image_name],...] num=32739
    with open(annotation_file) as json_file:
        label_file = json.load(json_file)

        annotation_list = []

        for i in range(len(label_file)):
            annotation = [int(label_file[i]["label_id"]),
                              label_file[i]["image_id"]]
            annotation_list.append(annotation)

    return annotation_list


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


def _process_image_worker(tup):
    process, img, train = tup
    ret = process(img, train)
    return ret


def func_batch_handle_with_multi_process(batch_x, train, standard):
    '''batch_x: PIL image list'''
    if train:
        pool = multiprocessing.Pool()
        # result = pool.map(
        #     _process_image_worker,
        #     ((aug_images_single, image) for image in batch_x)
        # )
        result = pool.map(_process_image_worker,
                          ((aug_images_single, image, train) for image in batch_x))
        pool.close()
        pool.join()
        batch_x = np.array(result)
    else:
        pool = multiprocessing.Pool()

    
    if standard:
        batch_x = z_score(batch_x)
    else:
        # batch_x = batch_x.astype(np.float32)
        batch_x = batch_x / 127.5
        batch_x = batch_x - 1
    return batch_x


def func_batch_handle(batch_x, train, standard):
    if train:
        image_list = []
        for image in batch_x:
            image = aug_images_single(image)
            image_list.append(image)
        batch_x = np.array(image_list)



def data_generator(annotation_list, batch_size, image_size,
                   image_dir, train, standard, crop_mode):
    '''data generator for fit_generator'''
    n = len(annotation_list)
    i = 0
    while True:
        image_data = []
        label_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_list)
            image, label = load_data(annotation_list[i],
                                     image_size, image_dir, crop_mode)
            image_data.append(image)
            label_data.append(label)
            i = (i+1) % n

        #多进程数据增强
        image_data = func_batch_handle_with_multi_process(image_data, train,
                                                          standard)

        label_data = np.array(label_data)
        yield (image_data, label_data)


def data_generator_wrapper(annotation_list, batch_size, image_size, image_dir,
                           train=True, standard=True, crop_mode='random'):
    n = len(annotation_list)
    if n==0 or batch_size<=0:
        return None
    return data_generator(annotation_list, batch_size, image_size,
                          image_dir, train, standard, crop_mode)
























