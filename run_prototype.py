import sys
import os
from resize_img import duplicate_resized_directory
from generate_new_images import generate_new_images
from cnn import create_cnn
from cnn_image_generator import create_cnn

size = [256]
channels = [3]
prototype = 5
PATH = os.getcwd()
num_classes = 10
num_epoch = 20
batch_size = 16
dataset_size=10000
train_fraction=0.65
validation_fraction=0.35

prototype_folder = os.path.join(PATH, 'Prototype_' + str(prototype))

if not os.path.exists(prototype_folder):
    os.makedirs(prototype_folder)

for s in size:
    for ch in channels:
        prototype_folder_iter = os.path.join(prototype_folder, 'Size_' + str(s) + '_Channels_' + str(ch))
        if not os.path.exists(prototype_folder_iter):
            os.makedirs(prototype_folder_iter)
        f = open(os.path.join(prototype_folder_iter, 'Prototype_info'), 'w+')
        f.write('Prototype number ' + str(prototype) + ' size=' + str(s) + 'x' + str(s) + ' channels=' + str(ch))
        f.close()
        sys.stdout = open(os.path.join(prototype_folder_iter, 'output'), "w")
        data_folder = os.path.join(os.path.join(PATH, 'Optimize_memory'), 'data_resized_' + str(s) + '_' + str(ch))
        """"'data_prototype_' + str(prototype) + '_' + str(s) + '_' + str(ch))
        if not os.path.exists(data_folder):
            duplicate_resized_directory(str(s), str(ch), data_folder)
            generate_new_images(data_folder)
        create_cnn(num_classes, data_folder, num_epoch, prototype_folder_iter, ch)"""
        create_cnn(num_classes=num_classes, data_path=data_folder, num_epoch=num_epoch,
                   path_to_store=prototype_folder_iter, channels=ch, img_rows=s, img_cols=s, batch_size=batch_size,
                   dataset_size=dataset_size, train_fraction=train_fraction, validation_fraction=validation_fraction)
        sys.stdout.close()
