from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import sys
import numpy as np
import glob
import sys

def image_genarator(trainimage,labelimage):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    """
    datagen = ImageDataGenerator(rotation_range=180,
                                 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode='nearest')
    image_data_len = glob.glob("img_folder/*")
    
    save_dir_train = "images_generated_train" 
    save_dir_label = "images_generated_label" 
    save_dir_list = [save_dir_train, save_dir_label]
    for i in save_dir_list:
        if not os.path.exists(i):
            os.mkdir(i)
    """
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    image_datagen.fit(trainimage, augment=True, seed=seed)
    mask_datagen.fit(labelimage, augment=True, seed=seed)
    #image_datagen.fit(images, augument=True, seed=seed)
    #mask_datagen.fit(masks, augment=True, seed=seed)
    traingenerator = image_datagen.flow(
        np.array(trainimage),
        batch_size=16
    )
    traingenerator_label = mask_datagen.flow(
        np.array(labelimage),
        batch_size=16
    )
    train_generator = zip(traingenerator, traingenerator_label)

    return train_generator
    
    
    
    """
    for i in range(len(image_data_len)):
        train_str = "img_folder/shot" + str(i + 1) + ".jpg"
        train_label = "label_folder/" + str(i + 1) + ".png"
        trainimg = cv2.imread(train_str)
        labelimg = cv2.imread(train_label)
        traingenerator = datagen.flow(
            np.array([trainimg]),
            batch_size=1
        )
        traingenerator_label = datagen.flow(
            np.array([labelimg]),
            batch_size=1
        )
        batches = traingenerator
        batches_label = traingenerator_label
        g_img = batches[0].astype(np.uint8)
        g_img_label = batches_label[0].astype(np.uint8)
        imagename = "images_" + str(i)  + ".png"
        imagename_label = "images_" + str(i) + "_" + "label" + ".png"
        output_dir = os.path.join(save_dir_list[0], imagename)
        output_dir_label = os.path.join(save_dir_list[1], imagename_label)
        cv2.imwrite(output_dir, g_img[0])
        cv2.imwrite(output_dir_label, g_img_label[0])
    """

