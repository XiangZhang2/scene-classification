# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from utils import get_json, data_generator_wrapper, unzip, label_shuffle
from utils import recycle_pool
from models.inception_resnet_v2 import build_model


def main(model_name = 'resnet50', continue_training=False):

  ucloud = False

  if model_name == 'resnet50':
    batch_size = 32
    image_size = 224
  elif model_name == 'inception_resnet_v2':
    batch_size = 16
    image_size = 299
  else:
    raise ValueError('model name uncorrect!!')
  

  if ucloud:
    log_dir = '/data/output/'
    train_annotation_path = '/data/data/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
    val_annotation_path = '/data/data/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    train_image_dir = '/data/data/ai_challenger_scene_train_20170904/scene_train_images_20170904/'
    val_image_dir = '/data/data/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'

    train_zip = '/data/data/ai_challenger_scene_train_20170904.zip'
    val_zip = '/data/data/ai_challenger_scene_validation_20170908.zip'
    zip_output = '/data/data/'

    unzip(train_zip, zip_output)
    unzip(val_zip, zip_output)

  else:
    log_dir = 'log/darknet/'
    train_annotation_path = '/Users/xiang/Desktop/classification/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
    val_annotation_path = '/Users/xiang/Desktop/classification/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    train_image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_train_20170904/scene_train_images_20170904/'
    val_image_dir = '/Users/xiang/Desktop/classification/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'

  if model_name == 'inception_resnet_v2':
    flag = 'ir2'
    if ucloud:
      weights = '/data/code/imagenet_models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    else:
      weights = 'imagenet_models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    # model = build_model(weights)
    # print('success creat inception_resnet_v2 model!!!')
    # #if continue tranning,modify this
    # #model.load_weights(weights_path)
    # print('load_weights succeed!!')

  
  if model_name == 'resnet50':
    flag = 'res50'
    if ucloud:
      weights = '/data/code/imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    else:
      weights = 'imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
  if model_name == 'densenet':
    flag = 'dense'
    print('success creat Densenet169 model!!!')
    print('load_weights succeed!!')

  if continue_training:
    weights = None
    weights_path = ''
    model = build_model(weights)
    print('success creat ' + str(model_name))
    model.load_weights(weights_path)
    print('load_weights succeed!!')
    
  model = build_model(weights)
  print('success creat ' + str(model_name))
  print('load_weights succeed!!')
  


  logging = TensorBoard(log_dir=log_dir)
  checkpoint = ModelCheckpoint(log_dir + flag + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
      monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

  #get train image path and image nums
  train_anno_list = get_json(train_annotation_path)
  #train_anno_list = label_shuffle(train_anno_list) #add label shuffling
  num_train = len(train_anno_list)
  val_anno_list = get_json(val_annotation_path)
  num_val = len(val_anno_list)

  
  #先训练最后三层（AVP、Desne、Dense）
  for layer in model.layers[:-3]:
    layer.trainable = False
  model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(data_generator_wrapper(train_anno_list, batch_size, image_size, train_image_dir, train=True),
          steps_per_epoch=max(1, num_train//batch_size),
          validation_data=data_generator_wrapper(val_anno_list, batch_size, image_size, val_image_dir, train=False, crop_mode='random'),
          validation_steps=max(1, num_val//batch_size),
          epochs=1,
          initial_epoch=0,
          callbacks=[logging, checkpoint])
  model.save_weights(log_dir + 'train_only_fc_2epoches.h5')
  recycle_pool()

  #训练所有层
  for layer in model.layers:
    layer.trainable = True
  model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(data_generator_wrapper(train_anno_list, batch_size, image_size, train_image_dir, train=True),
          steps_per_epoch=max(1, num_train//batch_size),
          validation_data=data_generator_wrapper(val_anno_list, batch_size, image_size, val_image_dir, train=False, crop_mode='random'),
          validation_steps=max(1, num_val//batch_size),
          epochs=10,
          initial_epoch=1,
          callbacks=[logging, checkpoint])
  model.save_weights(log_dir + 'train_all_layers_stage1.h5')
  recycle_pool()

  #继续训练
  for layer in model.layers:
    layer.trainable = True
  model.compile(optimizer=Adam(lr=5e-6), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(data_generator_wrapper(train_anno_list, batch_size, image_size, train_image_dir, train=True),
          steps_per_epoch=max(1, num_train//batch_size),
          validation_data=data_generator_wrapper(val_anno_list, batch_size, image_size, val_image_dir, train=False, crop_mode='random'),
          validation_steps=max(1, num_val//batch_size),
          epochs=8,
          initial_epoch=5,
          callbacks=[logging, checkpoint])
  model.save_weights(log_dir + 'train_all_layers_stage2.h5')
  recycle_pool()


if __name__ == "__main__":

    main()


