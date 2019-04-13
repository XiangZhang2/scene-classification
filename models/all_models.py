# -*- coding:utf-8 -*-
# third lib
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model

#own lib



def build_model(model_name='resnet50', weights=None):
    num_classes = 80
    if model_name == 'resnet50':
        base_model = ResNet50(include_top=False, weights=weights, input_shape=(224,224,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions, name='resnet50')

    elif model_name == 'inception_resnet_v2':
        base_model = InceptionResNetV2(include_top=False, weights=weights)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions, name='inception_resnet_v2')

    else:
        raise ValueError('model name uncorrect!!')
    return model

