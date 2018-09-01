"""
model_zoo.py
Assortment of CNN architectures for use with retinanet
@author: David Van Valen
"""

from keras_maskrcnn.models.retinanet import retinanet_mask

from keras.layers import Input
from keras_retinanet.models import retinanet
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation
from keras.models import Model
from deepcell.layers import ImageNormalization2D


def deepcell_backbone(inputs):
    """
    Retinanet uses feature pyramid networks(FPN) which combines features from
    different scales in the nueral network to obtain a better understanding
    of the image.

    Retinanet requires 3 features from the backbone called C3 , C4 and C5.

    Now the scaling here is as follows:
        If the model inputs are of the form -: Inputs : (x, x, 3)
        then  Cn = (x // (2^n), x // (2^n), None)
        Here a // b denotes rounding the float (a / b) to the largest integer.
    """
    
#     norm = ImageNormalization2D(norm_method='std')(inputs)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
   
    conv4 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='C3')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='C4')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Activation('relu')(conv5)
    pool5 = MaxPool2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='C5')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Activation('relu')(conv6)
    
    model = Model(inputs=inputs, outputs=conv6)

    return model


def deepcell_retinanet_mask(num_classes, backbone='deepcell', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using the custom backbone.
    # Args
        num_classes: Number of classes to predict.
        backbone: Our custom backbone.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone.
                  It could be used to freeze the training of the backbone.
    # Returns
        RetinaNet model with a custom backbone.
    """
    # choose default input
    if inputs is None:
        inputs = Input(shape=(None, None, 1))

    # Call the custom model
    deepcell_model = deepcell_backbone(inputs)
    # Make an array of the names of the layers we want
    layer_names = ['C3', 'C4', 'C5']
    # Get the required layers
    layer_outputs = [deepcell_model.get_layer(name).output for name in layer_names]
    backbone = Model(inputs=inputs, outputs=layer_outputs, name='deepcell')

    # create the full model
    return retinanet_mask(
        inputs=inputs,
        num_classes=num_classes,
        backbone_layers=backbone.outputs,
        **kwargs)
