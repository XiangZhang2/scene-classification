# -*- coding:utf-8 -*-
# standard lib

# third lib
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, ReduceLROnPlateau

# own lib
from models.all_models import build_model
from utils import get_json, unzip, label_shuffle
from data_augmentation import data_generator_wrapper


def main(model_name = 'resnet50', weights_mode='imagenet_no_top', on_server=True):
    """train the model.

    Args:
        model_name: one of ['resnet50', 'inception_resnet_v2', 'densenet169']
        weights_mode: load pretrained weights or not and use imagenet weights or own
            `None`: train from scratch
            `"imagenet_no_top"`: 
            `"pretrained"`: 
    """
    if model_name == 'resnet50':
        batch_size = 32
        image_size = 224
    elif model_name == 'inception_resnet_v2':
        batch_size = 16
        image_size = 299
    else:
        raise ValueError('model name uncorrect!!')

    if on_server:
        log_dir = '/data/output/'
        train_annotation_path = ('/data/data/ai_challenger_scene_train_20170904/'
                                'scene_train_annotations_20170904.json')
        val_annotation_path = ('/data/data/ai_challenger_scene_validation_20170908/'
                              'scene_validation_annotations_20170908.json')
        train_image_dir = ('/data/data/ai_challenger_scene_train_20170904/'
                          'scene_train_images_20170904/')
        val_image_dir = ('/data/data/ai_challenger_scene_validation_20170908/'
                        'scene_validation_images_20170908/')

        train_zip = '/data/data/ai_challenger_scene_train_20170904.zip'
        val_zip = '/data/data/ai_challenger_scene_validation_20170908.zip'
        zip_output = '/data/data/'

        unzip(train_zip, zip_output)
        unzip(val_zip, zip_output)

    else:
        log_dir = 'log/darknet/'
        train_annotation_path = ('/Users/xiang/Desktop/classification/'
                                'ai_challenger_scene_train_20170904/'
                                'scene_train_annotations_20170904.json')
        val_annotation_path = ('/Users/xiang/Desktop/classification/'
                              'ai_challenger_scene_validation_20170908/'
                              'scene_validation_annotations_20170908.json')
        train_image_dir = ('/Users/xiang/Desktop/classification/'
                          'ai_challenger_scene_train_20170904/'
                          'scene_train_images_20170904/')
        val_image_dir = ('/Users/xiang/Desktop/classification/'
                        'ai_challenger_scene_validation_20170908/'
                        'scene_validation_images_20170908/')

    if model_name == 'inception_resnet_v2':
        flag = 'ir2'
        if on_server:
            weights = ('/data/code/imagenet_models/'
                'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
        else:
            weights = ('imagenet_models/'
                'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

    if model_name == 'resnet50':
        flag = 'res50'
        if on_server:
            pretrained_weights = ('/data/code/imagenet_models/'
                '')
            imagenet_no_top_weights = ('/data/code/imagenet_models/'
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        else:
            pretrained_weights = ('imagenet_models/')
            imagenet_no_top_weights = ('imagenet_models/'
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    if model_name == 'densenet':
        flag = 'dense'

    #build model and load weights
    # with tf.device('/cpu:0'):
    if weights_mode == None:
        model = build_model(model_name)
    elif weights_mode == 'imagenet_no_top':
        model = build_model(model_name, weights=imagenet_no_top_weights)
    elif weights_mode == 'pretrained':
        model = build_model(model_name)
        model.load_weights(pretrained_weights)
    else:
        raise ValueError('weigths_mode uncorrect!')

    #callbacks
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + flag + 
                                 ('ep{epoch:03d}-loss{loss:.3f}-val_loss'
                                  '{val_loss:.3f}-val_acc{val_acc:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=3, verbose=1)

    #get train image path and image nums
    train_anno_list = get_json(train_annotation_path)
    #train_anno_list = label_shuffle(train_anno_list) #add label shuffling
    num_train = len(train_anno_list)
    val_anno_list = get_json(val_annotation_path)
    num_val = len(val_anno_list)


    for layer in model.layers:
        layer.trainable = True
    #parallel_model = multi_gpu_model(model, gpus=4)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(data_generator_wrapper(train_anno_list, batch_size,
                                               image_size, train_image_dir,
                                               train=True),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=data_generator_wrapper(val_anno_list,
                                                   batch_size,image_size, 
                                                   val_image_dir,train=False,
                                                   crop_mode=None),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=1,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'train_only_fc_2epoches.h5')


    #训练所有层
    for layer in model.layers:
        layer.trainable = True
    #parallel_model = multi_gpu_model(model, gpus=4)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(data_generator_wrapper(train_anno_list, batch_size,
                                               image_size, train_image_dir,
                                               train=True),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=data_generator_wrapper(val_anno_list,
                                                               batch_size,
                                                               image_size,
                                                               val_image_dir,
                                                               train=False,
                                                               crop_mode=None),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=2,
                        initial_epoch=1,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'train_all_layers_stage1.h5')


  #继续训练
    for layer in model.layers:
        layer.trainable = True
    #parallel_model = multi_gpu_model(model, gpus=4)
    model.compile(optimizer=Adam(lr=5e-6), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(
            data_generator_wrapper(train_anno_list, batch_size, image_size, 
                                   train_image_dir, train=True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(val_anno_list, batch_size, 
                                                   image_size, val_image_dir,
                                                   train=False, crop_mode=None),
            validation_steps=max(1, num_val//batch_size),
            epochs=6,
            initial_epoch=5,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping]
                                )
    model.save_weights(log_dir + 'train_all_layers_stage2.h5')



if __name__ == "__main__":
    print('training details: ')
    main(on_server=False)


