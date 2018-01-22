import itertools
import os
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

"""PATH = os.getcwd()
data_path = PATH + '/data_resized_extended_5'
data_dir_list = os.listdir(data_path)
data_dir_list.sort()"""


def create_cnn(num_classes, data_path, num_epoch, path_to_store, channels):
    img_data = get_images(data_path, channels)
    labels = create_labels(img_data.shape[0], data_path)
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)
    # Shuffle the dataset
    x, y = shuffle(img_data, Y, random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=2)
    model = define_model(img_data[0].shape, num_classes)
    callbacks_list = define_callbacks(path_to_store)
    hist = model.fit(x_train, y_train, batch_size=1, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test),
                     callbacks=callbacks_list)

    save_model(model, path_to_store)
    plot_confusion_matrix(model, x_test, y_test, path_to_store)


def get_images(data_path, channels):
    data_dir_list = os.listdir(data_path)
    data_dir_list.sort()
    img_data_list = []
    for dataset in data_dir_list:
        img_list = os.listdir(data_path + '/' + dataset)
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
            if channels == 1:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            img_data_list.append(input_img)
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    # Normalization
    img_data /= 255
    if channels == 1:
        img_data = np.expand_dims(img_data, axis=4)
    print(img_data.shape)
    return img_data


def create_labels(num_of_samples, data_path):
    labels = np.ones(num_of_samples, dtype='int64')
    data_dir_list = os.listdir(data_path)
    data_dir_list.sort()
    l = 0
    actual = 0
    for dataset in data_dir_list:
        nfiles = len(os.listdir(data_path + '/' + dataset))
        for i in range(actual, actual + nfiles):
            labels[i] = l
        l = l + 1
        actual = actual + nfiles
    print(Counter(labels))
    return labels


def define_model(input_shape, num_classes):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    """model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))"""

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    return model


def define_callbacks(path):
    filename = path + '/model_train_new.csv'
    csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

    # early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

    filepath = path + "/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    callbacks_list = [csv_log,
                      # , early_stopping,
                      checkpoint]
    return callbacks_list


def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path, "model.h5"))
    model.save(os.path.join(path, 'model.hdf5'))
    print("Saved model to disk")


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

# Plotting the confusion matrix
def plot_confusion_matrix(model, x_test, y_test, path_to_store, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    """s = [f for f in os.listdir('.') if 'Best' in f]
    filename = s[0]
    loaded_model = load_model('model.hdf5')
    loaded_model.load_weights(filename)"""
    y_pred = model.predict_classes(x_test, batch_size=1)
    cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))
    np.set_printoptions(precision=2)

    target_names = ['class 0(Aquabona)', 'class 1(Bezoya)', 'class 2(Evian)', 'class 3(Font_Vella)',
                    'class 4(Lanjaron)',
                    'class 5(Nestle_Aquarel)', 'class 6(Solan_de_Cabras)', 'class 7(Veri)', 'class 8(Vichy_Catalan)',
                    'class 9(Viladrau)']

    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure()
    # plt.savefig(path_to_store+'/confusion_matrix.jpg')

# %%
# Saving and loading model and weights
from keras.models import load_model

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
