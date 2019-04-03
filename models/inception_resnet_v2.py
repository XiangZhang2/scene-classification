from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model

#from config import num_classes



def build_model(weights, add_fc=False, model_name='resnet50'):
    num_classes = 80
    if model_name == 'resnet50':
        base_model = ResNet50(include_top=False, weights=weights, input_shape=(224,224,3))
        x = base_model.output
        x = Flatten()(x)
        # if add_fc:
        #     x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions, name='resnet50')

    elif model_name == 'inception_resnet_v2':
        base_model = InceptionResNetV2(include_top=False, weights=weights)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        if add_fc:
            x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions, name='inception_resnet_v2')

    else:
        raise ValueError('model name uncorrect!!')
    return model



# def build_model_fc(weights):
#     num_classes = 80
#     base_model = InceptionResNetV2(include_top=False, weights=weights)
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     #x = Dense(1024, activation='relu')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions, name='inception_resnet_v2')
#     return model
