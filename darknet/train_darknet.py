# -*- coding:utf-8 -*-
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import darknet, darknet_body
from utils import get_json, get_random_data, data_generator_wrapper



def creat_model(classes, load_pretrained=True, freeze_body=2,
                weights_path = 'imagenet_models/darknet53.h5'):
    
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    darknet53 = darknet_body(image_input)

    if load_pretrained:
        darknet53.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print('load_weights succeed!!!')

    darknet_model = darknet(darknet53, classes)
    model = Model(image_input, darknet_model.output)

    return model



def _main():
    classes = 80
    input_shape = (256, 256)
    log_dir = 'log/darknet/'
    train_annotation_path = '/Users/xiang/Desktop/classification/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
    val_annotation_path = '/Users/xiang/Desktop/classification/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    #train_annotation_path = 'new_label/AgriculturalDisease_train_annotations.json'
    #val_annotation_path = 'new_label/AgriculturalDisease_validation_annotations.json'
    train_image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_train_20170904/scene_train_images_20170904/'
    val_image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'


    model = creat_model(classes=80, load_pretrained=True, weights_path = 'imagenet_models/darknet53.h5')
    print('success creat darknet model!!!')
    
    print('model.layers=' + str(model.layers))
    print('len=' + str(len(model.layers)))

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #解压服务器上的数据
    # if os.path.isdir(image_output + 'image_2/'):
    #     print('Dont untar!')
    # else:
    #     untar(tar_file, image_output)

    train_anno_list = get_json(train_annotation_path)
    num_train = len(train_anno_list)
    val_anno_list = get_json(val_annotation_path)
    num_val = len(val_anno_list)


    if True:
        # for i in range(len(model.layers) - 3):
        #     model.layers[i].trainable = False  #训练最后三层
        sgd = SGD(lr=5e-3, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        batch_size = 16
        model.fit_generator(data_generator_wrapper(train_anno_list, batch_size, input_shape, train_image_dir, is_dark=True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(val_anno_list, batch_size, input_shape, val_image_dir, is_dark=True),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')



if __name__ == "__main__":

    _main()
