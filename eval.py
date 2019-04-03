# -*- coding=utf-8 -*-
import os
import json
from PIL import Image
import numpy as np
from keras import backend as K

from utils import unzip 
from models.inception_resnet_v2 import build_model

def get_random_data(image_list, image_size, image_dir, is_aug):
    
    image = Image.open(image_dir + image_list)
    input_shape = (image_size, image_size)
    image = image.resize(input_shape)
    image = np.array(image)
    # image = image / 127.5
    # image = image - 1
    image = image / 255.

    return image

# model = build_model_fc(weights=None)
# weights_path = 'imagenet_models/scene-output-ep003-loss0.930-val_loss0.867.h5'
# model.load_weights(weights_path)
# image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_test_a_20180103/scene_test_a_images_20180103/'
# image_list = '0a02a8129d1298f2bed666bdb60e65a20abc67cf.jpg'
# image = get_random_data(image_list, 299, image_dir, False)
# image = np.expand_dims(image, axis=0)
# pred = model.predict(image)
# pred1 = np.argmax(pred, axis = 1)
# pred3 = np.argsort(pred)[0][::-1][:3]
# print pred
# print pred1
# print pred3
# print pred.shape



def creat_submit(image_dir, submit_file, image_size, batch_size, weights_path, is_aug=False):
    '''用训练好的权重生成submit文件'''
    model = build_model(weights=None, model_name='resnet50')
    model.load_weights(weights_path)
    print('success creat inception_resnet_v2 model!!!')

    '''处理测试图片，生成batch'''
    image_list = os.listdir(image_dir)
    try:
        index = image_list.index('.DS_Store')
        image_list = image_list[:index] + image_list[index+1:]
    except:
        image_list = image_list


    num = len(image_list)
    r = num % batch_size  #15
    n = num / batch_size  #309

    i = 0
    batch = 0
    preds = []
    for a in range(n):
        image_batch = []
        for b in range(batch_size):
            image = get_random_data(image_list[i], image_size, image_dir, is_aug)
            image = np.expand_dims(image, axis=0)
            image_batch.append(image)
            i = i + 1
        x = np.concatenate([image for image in image_batch])
        y = model.predict_on_batch(x)
        #修改这里就可以了
        #pred = np.argmax(y, axis = 1) # for top 1
        for m in range(y.shape[0]):
            pred = np.argsort(y)[m][::-1][:3]
            pred = pred.tolist()
            preds.append(pred)
        batch = batch + 1
        print('process %d batch!!' % batch)

    if r != 0:
        image_batch = []
        for c in range(r):
            image = get_random_data(image_list[i], image_size, image_dir, is_aug)
            image = np.expand_dims(image, axis=0)
            image_batch.append(image)
            i = i + 1
        x = np.concatenate([image for image in image_batch])
        y = model.predict_on_batch(x)

        #pred = np.argmax(y, axis = 1) # 
        for l in range(y.shape[0]):
            pred = np.argsort(y)[l][::-1][:3]
            pred = pred.tolist()
            preds.append(pred)
        print('process remains!!')


    submit = []
    for k in range(len(image_list)):
        item = {}
        item['image_id'] = image_list[k]
        item['label_id'] = preds[k]
        submit.append(item)

    with open(submit_file, 'w') as f:
        json.dump(submit, f)
        print('write submit succeed!')


def __load_data(submit_file, reference_file, result):
  # load submit result and reference result

    with open(submit_file, 'r') as file1:
        submit_data = json.load(file1)
    with open(reference_file, 'r') as file1:
        ref_data = json.load(file1)
    if len(submit_data) != len(ref_data):
        result['warning'].append('Inconsistent number of images between submission and reference data \n')
    submit_dict = {}
    ref_dict = {}
    for item in submit_data:
        submit_dict[item['image_id']] = item['label_id']
    for item in ref_data:
        ref_dict[item['image_id']] = int(item['label_id'])
    return submit_dict, ref_dict, result


def __eval_result(submit_dict, ref_dict, result):
    # eval accuracy

    right_count_top3 = 0
    right_count_top1 = 0
    for (key, value) in ref_dict.items():

        if key not in set(submit_dict.keys()):
            result['warning'].append('lacking image %s in your submission file \n' % key)
            print('warnning: lacking image %s in your submission file' % key)
            continue

        # if value == submit_dict[key]:
        if value in submit_dict[key][:3]:
            right_count_top3 += 1
        if value == submit_dict[key][0]:
            right_count_top1 += 1

    result['top3'] = str(float(right_count_top3)/max(len(ref_dict), 1e-5))
    result['top1'] = str(float(right_count_top1)/max(len(ref_dict), 1e-5))
    return result


def eval(ucloud = False, model_name='resnet50'):
    
    if model_name == 'resnet50':
        batch_size = 32
        image_size = 224
    if model_name == 'inception_resnet_v2':
        batch_size = 16
        image_size = 299
    
    if ucloud:
        image_dir_testA = '/data/data/ai_challenger_scene_test_a_20180103/scene_test_a_images_20180103/'
        image_dir_testB = '/data/data/ai_challenger_scene_test_b_20180103/scene_test_b_images_20180103/'
        weights_path = '/data/code/imagenet_models/scene-output-trained_weights_all_layers_1.h5'
        submit_file_testA = '/data/output/testA_submit.json'
        submit_file_testB = '/data/output/testB_submit.json'
        reference_file_testA = '/data/data/ai_challenger_scene_test_a_20180103/scene_test_a_annotations_20180103.json'
        reference_file_testB = '/data/data/ai_challenger_scene_test_b_20180103/scene_test_b_annotations_20180103.json'

        testA_zip = '/data/data/ai_challenger_scene_test_a_20180103.zip'
        testB_zip = '/data/data/ai_challenger_scene_test_b_20180103.zip'
        zip_output = '/data/data/'

        unzip(testA_zip, zip_output)
        unzip(testB_zip, zip_output)

        creat_submit(image_dir_testA, submit_file_testA, image_size, batch_size, weights_path)
        creat_submit(image_dir_testB, submit_file_testB, image_size, batch_size, weights_path)

        result_A = {'error': [], 'warning': [], 'top3': None, 'top1': None}
        submit_dict_A, ref_dict_A, result_A = __load_data(submit_file_testA, reference_file_testA, result_A)
        result_A = __eval_result(submit_dict_A, ref_dict_A, result_A)

        result_B = {'error': [], 'warning': [], 'top3': None, 'top1': None}
        submit_dict_B, ref_dict_B, result_B = __load_data(submit_file_testB, reference_file_testB, result_B)
        result_B = __eval_result(submit_dict_B, ref_dict_B, result_B)

        print 'testA result =', result_A
        print 'testB result =', result_B

    else:
        #image_dir = '/Users/xiang/Desktop/classification/test/'
        image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_test_a_20180103/scene_test_a_images_20180103/'
        weights_path = 'imagenet_models/scene-output-res50ep008-loss0.981-val_loss0.779-val_acc0.767.h5'
        submit_file = 'submit/testA_res50.json'
        reference_file = '/Users/xiang/Desktop/classification/ai_challenger_scene_test_a_20180103/scene_test_a_annotations_20180103.json'

        creat_submit(image_dir, submit_file, image_size, batch_size, weights_path)

        result = {'error': [], 'warning': [], 'top3': None, 'top1': None}
        submit_dict, ref_dict, result = __load_data(submit_file, reference_file, result)
        result = __eval_result(submit_dict, ref_dict, result)
        print result


if __name__ == "__main__":
    eval()

