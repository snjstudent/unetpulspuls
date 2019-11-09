#import plaidml.keras
#plaidml.keras.install_backend()
import keras
from keras.layers.core import Activation
from keras.layers import Input, Concatenate, UpSampling2D, Reshape, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.layers.core import Activation, Dropout
from keras import backend, optimizers
import glob
import cv2
import sys
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import keras.backend as K
from data_augumetion import image_genarator




def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


#Vgg16を含めたモデル実装
#引用先:http://blog.neko-ni-naritai.com/entry/2018/04/07/115504
class Unetplusplus():
    def __init__(self, input_image):
        self.input_image = input_image

    def constract_model(self, input_image):
        filters_num = [16 * (2 ** i) for i in range(5)]
        print(filters_num)
        # inputs = Input(shape=(224, 224, 3))
        print(input_image[0].shape)
        inputs = Input(
            shape=(input_image.shape[1], input_image.shape[2], input_image.shape[3]))
        #print(input_image.shape)
        #hidden = Reshape((input_image.shape[0], input_image.shape[1],input_image.shape[2]), input_shape=(
        #    input_image.shape[0], input_image.shape[1]))(inputs)
        # Due to memory limitation, images will resized on-the-fly.
        x = Conv2D(64, (3, 3), activation='relu',
                   padding='same', name='block1_conv1')(inputs)
        #x = Dropout(0.3)(x)
        x1 = Conv2D(64, (3, 3), activation='relu',
                    padding='same', name='block1_conv2')(x)
        x = Dropout(0.3)(x1)
        x = MaxPooling2D((2, 2), strides=(
            2, 2), padding='same', name='block1_pool')(x)
        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='block2_conv1')(x)
        #x = Dropout(0.3)(x)
        x2 = Conv2D(128, (3, 3), activation='relu',
                    padding='same', name='block2_conv2')(x)
        x = Dropout(0.3)(x2)
        x = MaxPooling2D((2, 2), strides=(
            2, 2), padding='same', name='block2_pool')(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv1')(x)
        #x = Dropout(0.3)(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv2')(x)
        #x = Dropout(0.3)(x)
        x3 = Conv2D(256, (3, 3), activation='relu',
                    padding='same', name='block3_conv3')(x)
        x = Dropout(0.3)(x3)
        x = MaxPooling2D((2, 2), strides=(
            2, 2), padding='same', name='block3_pool')(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv1')(x)
        #x = Dropout(0.3)(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv2')(x)
        #x = Dropout(0.3)(x)
        x4 = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block4_conv3')(x)
        x = Dropout(0.3)(x4)
        x = MaxPooling2D((2, 2), strides=(
            2, 2), padding='same', name='block4_pool')(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv1')(x)
        #x = Dropout(0.3)(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv2')(x)
        #x = Dropout(0.3)(x)
        x5 = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(
            2, 2), padding='same', name='block5_pool')(x)
        x01 = Upsample2D_block(filters_num[0], 0, 1, skip=x1)(x2)
        x11 = Upsample2D_block(filters_num[1], 1, 1, skip=x2)(x3)
        x21 = Upsample2D_block(filters_num[2], 2, 1, skip=x3)(x4)
        x31 = Upsample2D_block(filters_num[3], 3, 1, skip=x4)(x5)
        x02 = Upsample2D_block(filters_num[0], 0, 2, skip=[x1, x01])(x11)
        x12 = Upsample2D_block(filters_num[1], 1, 2, skip=[x2, x11])(x21)
        x22 = Upsample2D_block(filters_num[2], 2, 2, skip=[x3, x21])(x31)
        x03 = Upsample2D_block(filters_num[0], 0, 3, skip=[x1, x02])(x12)
        x13 = Upsample2D_block(filters_num[1], 1, 3, skip=[x2, x12])(x22)
        x04 = Upsample2D_block(filters_num[0], 0, 4, skip=[x1, x03])(x13)
        x04 = Conv2D(3, (3, 3), padding="same")(x04)
        x04 = Activation('relu')(x04)
        #x04 = Upsample2D_block(3, 0, 4, skip=[x1, x03])(x13)
        model = Model(inputs=[inputs], outputs=[x04])
        return model


def handle_block_names(stage, cols):
    conv_name = 'decoder_stage{}-{}_conv'.format(stage, cols)
    bn_name = 'decoder_stage{}-{}_bn'.format(stage, cols)
    relu_name = 'decoder_stage{}-{}_relu'.format(stage, cols)
    up_name = 'decoder_stage{}-{}_upsample'.format(stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    return conv_name, bn_name, relu_name, up_name, merge_name


def ConvRelu(filters, kernel_size, use_batchnorm=True, use_dropout=True, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same",
                   name=conv_name, use_bias=not (use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        if use_dropout:
            x = Dropout(0.35)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def Upsample2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2),
                     use_batchnorm=True, skip=None):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(
            stage, cols)
        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                x = Concatenate(name=merge_name)([x] + skip)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer


def load_traindata(img_list, label_list):
    x = [image.load_img(i, target_size=(512, 256, 3)) for i in img_list]
    x = [image.img_to_array(i) for i in x]
    x = [np.array(preprocess_input(i))/255.0 for i in x]
    x = [np.expand_dims(i, axis=0)for i in x]
    y = [image.load_img(i, target_size=(512, 256, 3)) for i in label_list]
    y = [image.img_to_array(i) for i in y]
    y = [np.array(preprocess_input(i)) / 255.0 for i in y]
    y = [np.expand_dims(i, axis=0) for i in y]
    return x, y


def train_network(epochs):
    n_img_list = glob.glob('img_folder/*')
    img_list, label_list = [], []
    for i in range(len(n_img_list)):
        str_img, str_label = "img_folder/shot" + \
            str(i + 1) + ".jpg", "label_folder/" + str(i + 1) + ".png"
        img_list.append(str_img)
        label_list.append(str_label)
    x, y = load_traindata(img_list, label_list)
    models = Unetplusplus(x[0]).constract_model(x[0])
    #previous learing late = 3e-5
    adam = optimizers.Adam(lr=3e-4)
    models.compile(optimizer=adam, loss=bce_dice_loss)
    try:
        models.load_weights("segmentationss_3e4.hdf5")
    except:
        pass
    for i in range(epochs):
        if i % 50 == 0 and i != 0:
          models.save_weights("segmentationss_3e4.hdf5")
          from pydrive.auth import GoogleAuth
          from pydrive.drive import GoogleDrive
          from google.colab import auth
          from oauth2client.client import GoogleCredentials
          auth.authenticate_user()
          gauth = GoogleAuth()
          gauth.credentials = GoogleCredentials.get_application_default()
          drive = GoogleDrive(gauth)
          upload_file = drive.CreateFile()
          upload_file.SetContentFile("segmentationss_3e4.hdf5")
          upload_file.Upload()
        #print(image_genarators)
        #sys.exit()
        #print(image_genarators[1].shape)
        #sys.exit()
        if i == 1 or i % 10 == 0 and i > 0:
            testimage = models.predict(x[u], batch_size=16)
            print(testimage[0].max())
            print(testimage[0].max() * 255.0)
            cv2.imwrite("test.png", np.array(testimage[0]) * 255.0)
            print("")
            print("")
            print("image created")
            print("")
            print("")
        np.random.seed(1)
        np.random.shuffle(x)
        np.random.seed(1)
        np.random.shuffle(y)
        for u in range(len(x)):
            testimage = models.predict(x[u], batch_size=16)
            print(testimage[0].max())
            print(testimage[0].max() * 255.0)
            cv2.imwrite("test.png", np.array(testimage[0]) * 255.0)
            image_genarators = image_genarator(x[u], y[u])
            #loss += bce_dice_loss(backend.constant(y[i]), backend.constant(genarated_image))
            models.fit_generator(image_genarators, steps_per_epoch=5, epochs=1)
            #models.fit(x[u], y[u], batch_size=16)
            #testimage=testimage.clip(0,1)


train_network(50000)


def Transpose2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2),
                      transpose_kernel_size=(4, 4), use_batchnorm=False, skip=None):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(
            stage, cols)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name)(merge_list)
            else:
                x = Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer
