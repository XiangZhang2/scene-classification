# coding=utf-8
# standard lib
import os
import json
import random
import zipfile
import multiprocessing

# third lib
import numpy as np
# import matplotlib.pyplot as plt  # in ucloud
from tensorflow.python.client import device_lib

# own lib

def get_available_gpus():
    """get GPU number"""
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    """get the number of CPU"""
    return multiprocessing.cpu_count()



def get_class(class_file):
    """"""
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
    """类别平衡策略，将每个类别的图片数量扩充到和最大图片数量一致

    Args: annotation_list like [['label1_', 'image_2'],['label2_', 'image_2']···]
    returns: annotation_list_with_shuffle with same style
    """
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


