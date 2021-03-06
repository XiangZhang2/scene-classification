# -*- coding:utf-8 -*-
# standard lib
import random
import multiprocessing
from fractions import Fraction

# third lib
import numpy as np
from PIL import Image, ImageEnhance
#import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# own lib



def ten_crop(image, crop_mode='center'):
    # five crop，加翻转可实现ten crop
    image = image.resize((500, 500), Image.ANTIALIAS)
    box = {'left_up': (0, 0, 299, 299),
           'left_down': (0, 201, 299, 500),
           'center': (100, 100, 399, 399),
           'right_up': (201, 0, 500, 299),
           'right_down': (201, 201, 500, 500)}
    # for k in range(len(box)):
    #     crop = change.crop(box[k]['coord'])
    #print box[crop_mode]
    image = image.crop(box[crop_mode])

    return image


def image_resize(image, size):
    #先等比例缩小，把一边缩到299
    w = np.shape(np.array(image))[1]
    h = np.shape(np.array(image))[0]
    w = float(w)
    h = float(h)
    #print w,h,w/h
    if (w <= h and w == size) or (h <= w and h == size):
        return image
    if w < h:
        #print int(h/w*size)
        return image.resize((size, int(h/w * size)), Image.BICUBIC)
    else:
        #print int(w/h*size)
        return image.resize((int(w/h * size), size), Image.BICUBIC)


def center_crop(image, size):
    #单纯的在中间位置crop一个（size,size）的图片
    w = np.shape(np.array(image))[1]
    h = np.shape(np.array(image))[0]
    x1 = np.ceil((w - size) / 2)
    y1 = np.ceil((h - size) / 2)
    box = (x1, y1, x1 + size, y1 + size)
    c_crop = image.crop(box) #crop函数：如果长宽不够会用黑色填充，试下一下随机插值？
                             #可以先判断是否<299，填充以后再crop
    return c_crop


def random_crop(image, size):
    
    w = np.shape(np.array(image))[1]
    h = np.shape(np.array(image))[0]
    area = w * h
    target_area = np.random.uniform(0.08, 1) * area
    aspect_ratio = np.random.uniform(Fraction(3, 4), Fraction(4, 3))
    w_ = np.round(np.sqrt(target_area * aspect_ratio))
    h_ = np.round(np.sqrt(target_area / aspect_ratio))

    if h_ <= np.shape(np.array(image))[0] and w_ <= np.shape(np.array(image))[1]:
        x1 = np.random.uniform(0, w - w_)
        y1 = np.random.uniform(0, h - h_)
        box = (x1, y1, x1 + w_, y1 + h_)
        crop_image = image.crop(box)
        # plt.imshow(crop_image)
        # plt.show()
        scale = image_resize(crop_image, size)
        # plt.imshow(scale)
        # plt.show()
        image = center_crop(scale, size)
        # plt.imshow(c_crop)
        # plt.show()
        return image
    else:
        image = image.resize((size, size), Image.BICUBIC)
        return image


def z_score(image):
    '''标准化预处理'''
    #image = image.astype(np.float32)
    scale = 1 / 255.
    image = image * scale

    mean = [0.4960301824223457, 0.47806493084428053, 0.44767167301470545]
    var = [0.084966025569294362, 0.082005493489533315, 0.088877477602068156]
    
    if len(image.shape) == 3:
        image = image.astype(np.float32)
        image[:,:,0] = (image[:,:,0] - mean[0]) / var[0]
        image[:,:,1] = (image[:,:,1] - mean[1]) / var[1]
        image[:,:,2] = (image[:,:,2] - mean[2]) / var[2]
    elif len(image.shape) == 4:
        image[:,:,:,0] = (image[:,:,:,0] - mean[0]) / var[0]
        image[:,:,:,1] = (image[:,:,:,1] - mean[1]) / var[1]
        image[:,:,:,2] = (image[:,:,:,2] - mean[2]) / var[2]
    else:
        raise Exception('Format error.')

    return image


def color_jitter(image):
    '''颜色增强'''
    brightness_var = 0.4
    saturation_var = 0.5
    contrast_var = 0.4
    sharpness_var = 0.4

    alpha = 1 + random.uniform(-brightness_var, brightness_var)
    beta = 1 + random.uniform(-saturation_var, saturation_var)
    gamma = 1 + random.uniform(-contrast_var, contrast_var)
    deta = 1 + random.uniform(-sharpness_var, sharpness_var)

    image = ImageEnhance.Brightness(image).enhance(alpha)
    image = ImageEnhance.Color(image).enhance(beta)
    image = ImageEnhance.Contrast(image).enhance(gamma)
    image = ImageEnhance.Sharpness(image).enhance(deta)

    return image


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3,
                      r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def load_data(annotation_list, image_size, image_dir, crop_mode=None):
    '''load image and label,image:PIL format'''
    image = Image.open(image_dir + annotation_list[1])

    if crop_mode == None:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    elif crop_mode == 'random':
        image = random_crop(image, image_size)
    else:
        raise Exception('The crop_mode does not exist.')

    label = np.array(annotation_list[0])
    label = to_categorical(label, num_classes=80)  #modify there if classes is not 80

    return image, label


def load_and_process_single_image(image_path, image_size, train, crop_mode):
    '''image: PIL image, return a np array
    Args:
        image_path:
        image_size:
        train:
        crop_mode
    returns:
        image: numpy array
    '''
    image = Image.open(image_path)
    if crop_mode == "random":
        image = random_crop(image, image_size)
    elif crop_mode == None:
        image = image.resize((image_size, image_size))
    else:
        raise ValueError('Unknown crop_mode')

    if train:
        image = color_jitter(image)
        
        #random flip left and right
        if np.random.uniform() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #random erasing
        image = np.asarray(image)
        image.flags.writeable = True
        eraser = get_random_eraser()
        image = eraser(image)
    else:
        image = np.asarray(image)
        image.flags.writeable = True

    return image


def _process_image_worker(tup):
    process, img_path, image_size, train, crop_mode = tup
    ret = process(img_path, image_size, train, crop_mode)
    return ret


def batch_handle_with_multi_process(image_list, image_size, train,
                                    crop_mode, standard):
    '''batch_x: image_path '''
    pool = multiprocessing.Pool()
    result = pool.map(_process_image_worker,
        ((load_and_process_single_image, image_path, 
          image_size, train, crop_mode) for image_path in image_list))
    pool.close()
    pool.join()
    batch_x = np.array(result)

    if standard:
        batch_x = z_score(batch_x)
    else:
        # batch_x = batch_x.astype(np.float32)
        batch_x = batch_x / 127.5
        batch_x = batch_x - 1
    return batch_x


def batch_handle(image_list, image_size, train, crop_mode, standard):
    batch_x = []
    for image_path in image_list:
        image = load_and_process_single_image(image_path, image_size,
                                              train, crop_mode)
        batch_x.append(image)
    batch_x = np.array(batch_x)

    if standard:
        batch_x = z_score(batch_x)
    else:
        # batch_x = batch_x.astype(np.float32)
        batch_x = batch_x / 127.5
        batch_x = batch_x - 1
    return batch_x


def data_generator(annotation_list, batch_size, image_size,
                   image_dir, train, standard, crop_mode, multi_process):
    '''data generator for fit_generator'''
    n = len(annotation_list)
    i = 0
    while True:
        batch_list = [] #add
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_list)
            batch_list.append(annotation_list[i]) #add
            i = (i+1) % n

        image_list = [image_dir+batch_list[j][1] for j in range(len(batch_list))]
        label_list = [batch_list[j][0] for j in range(len(batch_list))]

        #多进程数据增强
        if multi_process:
            image_data = batch_handle_with_multi_process(image_list, 
                                        image_size, train, crop_mode, standard)
        else:
            image_data = batch_handle(image_list, image_size,
                                      train, crop_mode, standard)

        label_data = [to_categorical(label, num_classes=80) 
                      for label in label_list]
        label_data = np.array(label_data)
        yield (image_data, label_data)


def data_generator_wrapper(annotation_list, batch_size, image_size, image_dir,
                           train=True, standard=True, crop_mode='random', 
                           multi_process=False):
    n = len(annotation_list)
    if n==0 or batch_size<=0:
        return None
    return data_generator(annotation_list, batch_size, image_size,
                          image_dir, train, standard, crop_mode, multi_process)


