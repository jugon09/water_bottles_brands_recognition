import cv2
import os

from keras import backend as K

K.set_image_dim_ordering('tf')
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np


def generate_new_images(data_path_resized):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2,
        vertical_flip=True,
        fill_mode='nearest')
    c = 0
    data_dir_list = os.listdir(data_path_resized)
    data_dir_list.sort()
    for dataset in data_dir_list:
        final_directory = os.path.join(data_path_resized, dataset)
        img_list = os.listdir(final_directory)
        labels = np.full(shape=(len(img_list)), fill_value=c, dtype='int64')
        print('Loaded the images of resized dataset-' + '{}\n'.format(dataset))
        img_list_ready = []
        for img in img_list:
            input_img = cv2.imread(final_directory + '/' + img)
            # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge((r, g, b))
            img_list_ready.append(input_img)
        img_list_ready = np.array(img_list_ready)
        img_list_ready = img_list_ready.astype('float32')
        i = 0
        for batch in datagen.flow(img_list_ready, labels, batch_size=1,
                              save_to_dir=final_directory, save_prefix='generated',
                              save_format='jpeg'):
            i = i + 1
            if i >= 1000 - len(img_list):
                break  # otherwise the generator would loop indefinitely
        c = c + 1


#PATH = os.getcwd()

# data_path_resized = PATH + '/data_resized_extended_7'
