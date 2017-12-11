import cv2
import os
import shutil


def duplicate_resized_directory(size, channels, new_folder):
    source_folder = os.getcwd() + '/data_resized_' + size + '_' + channels
    shutil.copytree(source_folder, new_folder)

"""PATH = os.getcwd()
folder_to_store = PATH + '/data_resized_128_3'
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)
data_dir_list.sort()

img_rows = 128
img_cols = 128

if not os.path.exists(folder_to_store):
    os.makedirs(folder_to_store)

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    final_directory = os.path.join(folder_to_store, dataset)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
        cv2.imwrite(final_directory + '/' + img + '_resize', input_img_resize)"""
