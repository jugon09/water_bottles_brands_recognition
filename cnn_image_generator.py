import itertools
import os
import csv
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

K.set_image_dim_ordering('tf')

from keras import optimizers
from keras import callbacks
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import cv2
from keras import backend as K

K.set_image_dim_ordering('tf')
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import model_from_json
from keras.models import load_model

"""PATH = os.getcwd()
data_path = PATH + '/data_resized_extended_5'
data_dir_list = os.listdir(data_path)
data_dir_list.sort()"""


def create_cnn(num_classes, data_path, num_epoch, path_to_store, channels, img_rows, img_cols, batch_size=32,
               dataset_size=10000, train_fraction=0.65, validation_fraction=0.35, optimizer='rmsprop', dropout=0.5,
               num_neurs=64):
    train_size = int(dataset_size * train_fraction)
    validation_size = int(dataset_size * validation_fraction)
    is_rgb = True if channels == 3 else False
    train_generator = create_train_generator(data_path=data_path, width=img_rows, height=img_cols,
                                             batch_size=batch_size, is_rgb=is_rgb)
    validation_generator = create_validation_generator(data_path=data_path, width=img_rows, height=img_cols,
                                                  batch_size=batch_size, is_rgb=is_rgb)
    model = define_model(input_shape=train_generator.image_shape, num_classes=num_classes, optimizer=optimizer,
                         dropout=dropout, num_neurs=num_neurs)
    callbacks_list = define_callbacks(path_to_store)
    model.fit_generator(
        train_generator,
        samples_per_epoch=train_size,
        epochs=num_epoch,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=validation_size)
    save_model(model, path_to_store)
    model = select_best_model(data_path=path_to_store, model=model, window_size=5)
    test_generator = create_test_generator(data_path=data_path, width=img_rows, height=img_cols,
                                                       batch_size=batch_size, is_rgb=is_rgb)
    plot_confusion_matrix(model, test_generator, validation_size, path_to_store)
    compute_correlation_loss_acc(path_to_store)


def create_train_generator(data_path, width=128, height=128, batch_size=32, is_rgb=False):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1. / 255)
    color_mode = 'rgb' if is_rgb else 'grayscale'
    train_generator = datagen.flow_from_directory(
        directory=os.path.join(data_path, 'Train'),
        target_size=(width, height),
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode='categorical')
    return train_generator


def create_validation_generator(data_path, width=128, height=128, batch_size=32, is_rgb=False):
    test_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1. / 255
    )
    color_mode = 'rgb' if is_rgb else 'grayscale'
    validation_generator = test_datagen.flow_from_directory(
        directory=os.path.join(data_path, 'Validation'),
        target_size=(width, height),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    return validation_generator


def create_test_generator(data_path, width=128, height=128, batch_size=32, is_rgb=False):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    color_mode = 'rgb' if is_rgb else 'grayscale'
    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(data_path, 'Validation'),
        target_size=(width, height),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    return test_generator


def define_model(input_shape, num_classes, optimizer, dropout, num_neurs):
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    """model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))"""

    model.add(Flatten())
    model.add(Dense(num_neurs))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def define_callbacks(path):
    filename = path + '/model_train_new.csv'
    csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

    # early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    filepath = path + "/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto',
                                           save_weights_only=True)

    callbacks_list = [csv_log,
                      # , early_stopping,
                      checkpoint]
    return callbacks_list


def save_model(model, path):
    # serialize model to JSON
    model_json = model.model.to_json()
    with open(os.path.join(path, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print("Saved architecture to disk")


names = ['Aquabona', 'Bezoya', 'Evian', 'Font_Vella', 'Lanjaron', 'Nestle_Aquarel', 'Solan_de_Cabras', 'Veri',
         'Vichy_Catalan', 'Viladrau']


"""data_dir_list_test = os.listdir(PATH + 'imagenes_restantes/')
data_dir_list_test.sort()

for dataset in data_dir_list_test:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (128, 128))
        test_image = np.array(input_img_resize)
        test_image = test_image.astype('float32')
        test_image /= 255
        print((model.predict(test_image)))
        print(model.predict_classes(test_image))

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)"""
# Testing a new image
"""test_image = cv2.imread(PATH + '/imagenes_restantes/Solan_de_Cabras/20171026_204220.jpg')
test_image = cv2.resize(test_image, (img_rows, img_cols))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)

# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))"""


def select_best_model(data_path, model, window_size=5):
    """
    This function load the model weights with the best val_loss and deletes the others from data folder
    :param data_path:
    :param window_size:
    :param model
    """
    files = [f for f in os.listdir(data_path) if 'Best' in f]
    files.sort()
    n = len(files)
    # val_loss = (min_val_loss, position_val_loss_min)
    val_loss = [float('+Infinity'), -1]
    val_loss_position = 6
    i = 2
    while i + 2 < n:
        print('iter ' + str(i))
        suma = 0.0
        j = i - 2
        cont = 0
        while j < n and cont < window_size:
            epoch_values = files[j].split('-')
            suma = suma + float(epoch_values[val_loss_position])
            j = j + 1
            cont = cont + 1
        print('media ' + str(suma / window_size))
        if suma / window_size < val_loss[0]:
            val_loss[0] = suma / window_size
            val_loss[1] = i
        i = i + 1
    print('mejor media ' + str(val_loss[0]))
    print('pos ' + str(val_loss[1]))
    model.load_weights(os.path.join(data_path, files[val_loss[1]]))
    for f in files:
        if f != files[val_loss[1]]:
            os.remove(os.path.join(data_path, f))
    return model


# Plotting the confusion matrix
def plot_confusion_matrix(model, validation_generator, steps, path_to_store, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    """s = [f for f in os.listdir('.') if 'Best' in f]
    filename = s[0]
    loaded_model = load_model('model.hdf5')
    loaded_model.load_weights(filename)"""
    Y_pred = model.predict_generator(generator=validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print(classification_report(validation_generator.classes, y_pred))
   

def compute_correlation_loss_acc(folder):
    train_acc = []
    train_loss = []
    val_loss = []
    val_acc = []
    train_acc_col = 1
    train_loss_col = 2
    val_acc_col = 3
    val_loss_col = 4
    with open(os.path.join(folder, 'model_train_new.csv'), 'r') as f:
    #with open(os.path.join(os.getcwd(), 'Prototype_8/Size_128_Channels_1_opt_adam_num_neurs_32/model_train_new.csv'),
              #'r') as f:
        reader = csv.reader(f)
        is_header = True
        for row in reader:
            if row:
                # Save header row.
                if is_header:
                    header = row
                    is_header = False
                else:
                    train_acc.append(float(row[train_acc_col]))
                    train_loss.append(float(row[train_loss_col]))
                    val_acc.append(float(row[val_acc_col]))
                    val_loss.append(float(row[val_loss_col]))
    print('')
    print('Correlation between train_' + header[train_acc_col] + ' train_' + header[train_loss_col])
    print(np.corrcoef(train_acc, train_loss))
    print('')
    print('Correlation between ' + header[val_acc_col] + ' ' + header[val_loss_col])
    print(np.corrcoef(val_acc, val_loss))


def test_one_image(image_path):
    with open('model.json', 'r') as f:
        arch = f.read()
    model = model_from_json(arch)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.load_weights('Best-weights.h5')
    input_img = cv2.imread(image_path)
    input_img_resize = cv2.resize(input_img, (128, 128))
    test_image = np.array(input_img_resize)
    test_image = test_image.astype('float32')
    test_image /= 255
    return model.predict_classes(test_image)

# serialize model to JSON
"""model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
model.save('model2.hdf5')
print("Saved model to disk")"""
# load json and create model
"""json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")

model.save('model2.hdf5')
loaded_model = load_model('model2.hdf5')"""
