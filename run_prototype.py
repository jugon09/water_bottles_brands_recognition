import sys
import os
from resize_img import duplicate_resized_directory
from generate_new_images import generate_new_images
from cnn import create_cnn

size = [128, 256]
channels = [1, 3]
prototype = 2
PATH = os.getcwd()
num_classes = 10
num_epoch = 20

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
        data_folder = os.path.join(prototype_folder_iter,
                                   'data_prototype_' + str(prototype) + '_' + str(s) + '_' + str(ch))
        if not os.path.exists(data_folder):
            duplicate_resized_directory(str(s), str(ch), data_folder)
            generate_new_images(data_folder)
        create_cnn(num_classes, data_folder, num_epoch, prototype_folder_iter, ch)
        sys.stdout.close()
