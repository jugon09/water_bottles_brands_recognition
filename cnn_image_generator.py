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
               dataset_size=10000, train_fraction=0.65, validation_fraction=0.35):
    """img_data = get_images(data_path, channels)
    labels = create_labels(img_data.shape[0], data_path)
    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)
    # Shuffle the dataset
    x, y = shuffle(img_data, Y, random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=2)
    model = define_model(img_data[0].shape, num_classes)
    callbacks_list = define_callbacks(path_to_store)
    hist = model.fit(x_train, y_train, batch_size=4, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test),
                     callbacks=callbacks_list)
    """
    train_size = int(dataset_size * train_fraction)
    validation_size = int(dataset_size * validation_fraction)
    is_rgb = True if channels == 3 else False
    train_generator = create_train_validation(data_path=data_path, width=img_rows, height=img_cols,
                                              batch_size=batch_size, is_rgb=is_rgb)
    validation_generator = create_test_validation(data_path=data_path, width=img_rows, height=img_cols,
                                                  batch_size=batch_size, is_rgb=is_rgb)
    model = define_model(input_shape=train_generator.image_shape, num_classes=num_classes)
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
    plot_confusion_matrix(model, validation_generator, validation_size, path_to_store)
    compute_correlation_val_loss_acc_loss(path_to_store)


def create_train_validation(data_path, width=128, height=128, batch_size=32, is_rgb=False):
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


def create_test_validation(data_path, width=128, height=128, batch_size=32, is_rgb=False):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    color_mode = 'rgb' if is_rgb else 'grayscale'
    validation_generator = test_datagen.flow_from_directory(
        directory=os.path.join(data_path, 'Validation'),
        target_size=(width, height),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    return validation_generator


def define_model(input_shape, num_classes):
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
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
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
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


"""img_rows = 256
img_cols = 256
num_channel = 1
num_epoch = 20

num_classes = 10"""

"""for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        img_data_list.append(input_img)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
# Normalization
img_data /= 255"""

# %%
"""USE_SKLEARN_PREPROCESSING = False

if USE_SKLEARN_PREPROCESSING:
    # using sklearn for preprocessing
    from sklearn import preprocessing


    def image_to_feature_vector(image, size=(128, 128)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, size).flatten()


    img_data_list = []
    for dataset in data_dir_list:
        img_list = os.listdir(data_path + '/' + dataset)
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_flatten = image_to_feature_vector(input_img, (128, 128))
            img_data_list.append(input_img_flatten)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    print(img_data.shape)
    img_data_scaled = preprocessing.scale(img_data)
    print(img_data_scaled.shape)

    print(np.mean(img_data_scaled))
    print(np.std(img_data_scaled))

    print(img_data_scaled.mean(axis=0))
    print(img_data_scaled.std(axis=0))

    if K.image_dim_ordering() == 'th':
        img_data_scaled = img_data_scaled.reshape(img_data.shape[0], num_channel, img_rows, img_cols)
        print(img_data_scaled.shape)

    else:
        img_data_scaled = img_data_scaled.reshape(img_data.shape[0], img_rows, img_cols, num_channel)
        print(img_data_scaled.shape)

    if K.image_dim_ordering() == 'th':
        img_data_scaled = img_data_scaled.reshape(img_data.shape[0], num_channel, img_rows, img_cols)
        print(img_data_scaled.shape)

    else:
        img_data_scaled = img_data_scaled.reshape(img_data.shape[0], img_rows, img_cols, num_channel)
        print(img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
    img_data = img_data_scaled
    """
# %%
# Assigning Labels

# Define the number of classes
# num_classes = 10

"""num_of_samples = img_data.shape[0]
labels = np.ones(num_of_samples, dtype='int64')

l = 0
actual = 0
for dataset in data_dir_list:
    nfiles = len(os.listdir(data_path + '/' + dataset))
    for i in range(actual, actual + nfiles):
        labels[i] = l
    l = l + 1
    actual = actual + nfiles

print(Counter(labels))"""

"""names = ['Aquabona', 'Bezoya', 'Evian', 'Font_Vella', 'Lanjaron', 'Nestle_Aquarel', 'Solan_de_Cabras', 'Veri',
         'Vichy_Catalan', 'Viladrau']"""

""""# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)"""
# Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory


# %%
# Defining the model

"""input_shape = img_data[0].shape

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))"""

"""model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))"""

"""model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))"""

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

# Viewing model_configuration

"""model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable"""

# %%
# Training
# hist = model.fit(X_train, y_train, batch_size=32, epochs=num_epoch, verbose=1, validation_split=0.2)

# hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)

# Training with callbacks
"""from keras import callbacks

filename = 'model_train_new.csv'
csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

# early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

filepath = "Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

callbacks_list = [csv_log,
                  # , early_stopping,
                  checkpoint]"""

"""hist = model.fit(x, y, batch_size=32, epochs=num_epoch, verbose=1, validation_split=0.2,
                 callbacks=callbacks_list)"""

# visualizing losses and accuracy
"""train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
# print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
"""
# %%

# Evaluating the model

"""score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])"""

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

# %%
"""
# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
    activations = get_activations([X_batch, 0])
    return activations


layer_num = 3
filter_num = 0

activations = get_featuremaps(model, int(layer_num), test_image)

print(np.shape(activations))
feature_maps = activations[0][0]
print(np.shape(feature_maps))

if K.image_dim_ordering() == 'th':
    feature_maps = np.rollaxis((np.rollaxis(feature_maps, 2, 0)), 2, 0)
print(feature_maps.shape)

fig = plt.figure(figsize=(16, 16))
plt.imshow(feature_maps[:, :, filter_num], cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num) + '.jpg')

num_of_featuremaps = feature_maps.shape[2]
fig = plt.figure(figsize=(16, 16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i + 1)
    # ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(feature_maps[:, :, i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')
"""


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
            print(epoch_values[val_loss_position])
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
    """cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))
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
    # plt.savefig(path_to_store+'/confusion_matrix.jpg')"""


def compute_correlation_val_loss_acc_loss(folder):
    val_loss = []
    val_acc = []
    val_acc_col = 3
    val_loss_col = 4
    with open(os.path.join(folder, 'model_train_new.csv'), 'r') as f:
        reader = csv.reader(f)
        is_header = True
        for row in reader:
            if row:
                # Save header row.
                if is_header:
                    header = row
                    is_header = False
                else:
                    val_acc.append(float(row[val_acc_col]))
                    val_loss.append(float(row[val_loss_col]))
    print(header)
    print(np.corrcoef(val_acc, val_loss))

    """ifile = open(os.path.join(folder, 'model_train_new.csv'))
    reader = csv.reader(ifile)
    rownum = 0
    for row in reader:
        # Save header row.
        if rownum == 0:
            header = row
        else:
            val_acc.append(row[val_acc_col])
            val_loss.append(row[val_loss_col])
    ifile.close()
    print(header)
    print(np.corrcoef(val_acc, val_loss))"""


# Compute confusion matrix
"""cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
# plt.figure()
# Plot normalized confusion matrix
# plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
# plt.figure()
plt.show()
"""
# %%
# Saving and loading model and weights


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
loaded_model = load_model('model2.hdf5')

test_path = PATH + '/imagenes_restantes'
data_dir_list_test = os.listdir(test_path)
data_dir_list_test.sort()

for dataset in data_dir_list_test:
    img_list = os.listdir(test_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(test_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (img_rows, img_rows))
        test_image = np.array(input_img_resize)
        test_image = test_image.astype('float32')
        test_image /= 255
        test_image = np.expand_dims(test_image, axis=0)
        print(loaded_model.predict_classes(test_image))"""