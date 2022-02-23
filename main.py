"""Aditional definitions/import"""
#from keras.utils import to_categorial
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

import os
import random

import numpy
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
# from sklearn.esemble import confusion_matrix
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as  ET
import glob
from pathlib import Path

import shutil
#IMAGE THINGS :p
from matplotlib import image
import pandas as pd
from sklearn.datasets import images
from sklearn.model_selection import train_test_split

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg



        ######PATHS

path_ANN = r'../archiwum/annotations' #annotations folder
path_IMG=r'../archiwum/images' #from main foldder



path_temp_test=r'../test/annotations' #annotation test path
path_temp_train= r'../train/annotations' #annotation train path

path_temp_test_i=r'../test/images' #test path to images
path_temp_train_i= r'../train/images' #train path to images

path_to_test=r'../test'
path_to_train = r'../train'


counter_rand=0

#4 classes
crosswalk = []
other=[]

#train and test
train = []
test = []

# #image data
# data_img=[]


# def old_extract_data(path):  #other way but out of date
#     """Extracting needed data from .xml file"""
#             #LIGHT THOUGHTS
#     #print(os.getcwd())
#     #l = list(os.listdir())
#     #print(l)
#
#     """NEW LISTS AND COUNTER+TOTAL CHECK"""
#     total = 0
#     cnt = 0
#     counterofxmls=0
#     ###empty lists
#     # traffic_lights = []
#     # crosswalk = []
#     # stop = []
#     # speedlimit = []
#
#     allxml=[]
#
#     for entry in os.scandir(path):
#         tree = ET.parse(entry)
#         root = tree.getroot()
#         # name=root.findall('object')
#         for filename in root.findall('object'):
#             name = filename.find('name').text
#             #print(name)
#             if name == 'trafficlight':
#                 traffic_lights.append(entry)
#             elif name == 'stop':
#                 stop.append(entry)
#             elif name == 'speedlimit':
#                 speedlimit.append(entry)
#             elif name == 'crosswalk':
#                 crosswalk.append(entry)
#             cnt += 1
#         # print(name)
#         #         cnt += 1
#         if entry.is_dir(follow_symlinks=False):
#             total += extract_data(entry.path)
#         else:
#             total += entry.stat(follow_symlinks=False).st_size
#
#         for nameroad in root.findall('annotations'):
#             filnm=nameroad.find('filename').text
#             allxml.append(filnm)
#             counterofxmls+=1
#             #print(filnm)
#     # print(total)
#     # print(cnt)
#     #print(counterofxmls)
#     #print(allxml)
#
#     #####PRINTS OF ALL TYPES (HMM CLASSES)
#
#     print('traffic_light', len(traffic_lights))
#     print('stop', len(stop))
#     print('speedlimit', len(speedlimit))
#     print('crosswalk', len(crosswalk))
#     print('filename', len(allxml))
#
#     return total


def get_file_list(root, file_type):
    return [os.path.join(directory_path, a) for directory_path, directory_name,
            files in os.walk(root) for a in files if a.endswith(file_type)] ##zczytuje kazdy folder


def new_load_data(path_annotations):
    annotations_path_list = get_file_list(path_annotations, '.xml')
    for a_path in annotations_path_list:
        root = ET.parse(a_path).getroot()
        if root.find("./object/name").text == 'crosswalk':
            crosswalk.append(a_path)
            print(crosswalk)
        else:
            other.append(a_path)


def extract_data(path_annotations, path_images): #tu zmienic
    """Extracting needed data from .xml file and fitting to image"""
    annotations_path_list = get_file_list(path_annotations, '.xml')
    annotations_list = []
    for a_path in annotations_path_list:
        root = ET.parse(a_path).getroot()
        annotations = {}
        annotations['filename'] = Path(str(path_images) + '/' + root.find("./filename").text)
        annotations['images']=cv2.imread(str(path_images) + '/' + root.find("./filename").text)
        annotations['width'] = root.find("./size/width").text
        annotations['height'] = root.find("./size/height").text
        annotations['class'] = root.find("./object/name").text
        #boundbox description
        annotations['xmin'] = int(root.find("./object/bndbox/xmin").text)
        annotations['ymin'] = int(root.find("./object/bndbox/ymin").text)
        annotations['xmax'] = int(root.find("./object/bndbox/xmax").text)
        annotations['ymax'] = int(root.find("./object/bndbox/ymax").text)
        annotations_list.append(annotations)


    return annotations_list


def train_test_list(): #dzielenie 3 do 1
    counter=0

    while len(crosswalk) != 0:
        index = random.randrange(0, len(crosswalk))
        if counter == 0:
            test.append(crosswalk[index])
        else:
            train.append(crosswalk[index])
        crosswalk.pop(index)
        counter = (counter + 1) % 4
    while len(other) != 0:
        index = random.randrange(0, len(other))
        if counter == 0:
            test.append(other[index])
        else:
            train.append(other[index])
        other.pop(index)
        counter = (counter + 1) % 4

           #CLEAR
    ## annotations clearing
    for clear in os.listdir(path_temp_train):
        os.unlink(os.path.join(path_temp_train,clear))
    for clear in os.listdir(path_temp_test):
        os.unlink(os.path.join(path_temp_test, clear))
     ## images clearing
    for clear in os.listdir(path_temp_train_i):
        os.unlink(os.path.join(path_temp_train_i, clear))
    for clear in os.listdir(path_temp_test_i):
        os.unlink(os.path.join(path_temp_test_i, clear))

    for i in train:
        shutil.copy(i,path_temp_train)
        base=os.path.basename(i)
        name=os.path.splitext(base)[0]
        #print(name)
        shutil.copy(path_IMG+'/'+name+'.png',path_temp_train_i)
    for i in test:
        shutil.copy(i, path_temp_test)
        base = os.path.basename(i)
        name = os.path.splitext(base)[0]
        shutil.copy(path_IMG+'/'+name + '.png',path_temp_test_i)

    return train,test

def learn_bovw(data):

    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['images'], None) ## tutaj cos moze nie grac
        kpts, desc = sift.compute(sample['images'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)


def extract_features(data):

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data: #data to nasza duza tabela
        kpts=sift.detect(sample['images'],None)
        descs=bow.compute(sample['images'],kpts) #deskryptory
        sample['desc']=descs
        # ------------------

    return data


def train_algorithm(data):

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    des=[]
    labels=[]
    for sample in data:
        if sample['desc'] is not None:
            des.append(sample['desc'].squeeze(0))
            labels.append(sample['class'])

    clf.fit(des,labels)

    # ------------------

    return clf


def draw_grid(images, n_classes, grid_size, h, w):

    image_all = np.zeros((h, w, 3), dtype=np.uint8)
    h_size = int(h / grid_size)
    w_size = int(w / n_classes)

    col = 0
    for class_id, class_images in images.items():
        for idx, cur_image in enumerate(class_images):
            row = idx

            if col < n_classes and row < grid_size:
                image_resized = cv2.resize(cur_image, (w_size, h_size))
                #cv2.imshow('resize',image_resized)
                image_all[row * h_size: (row + 1) * h_size, col * w_size: (col + 1) * w_size, :] = image_resized

        col += 1

    return image_all


def predict(rf, data):

    for sample in data:
        if sample['desc'] is not None:
            y_pred=rf.predict(sample['desc'])
            sample['class_pred']=y_pred[0]

    return data


def evaluate(data):

    all=0
    positive=0
    pred_labels=[]
    true_labels=[]

    for sample in data:
        if sample['desc'] is not None:
            pred_labels.append(sample['class_pred'])
            true_labels.append(sample['class'])

            if sample['class'] == sample['class_pred']:
                positive = positive + 1
            all=all+1
    print(positive/all)

    conf_matrix=confusion_matrix(true_labels,pred_labels)
    # print(conf_matrix)
    # ------------------

    # this function does not return anything
    return


def display(data):

    n_classes = 3

    corr = {}
    incorr = {}

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['class_pred'] == sample['class']:
                if sample['class_pred'] not in corr:
                    corr[sample['class_pred']] = []
                corr[sample['class_pred']].append(idx)
            else:
                if sample['class_pred'] not in incorr:
                    incorr[sample['class_pred']] = []
                incorr[sample['class_pred']].append(idx)

            # print('ground truth = %s, predicted = %s' % (sample['label'], pred))
            # cv2.imshow('image', sample['image'])
            # cv2.waitKey()

    grid_size = 8

    # sort according to classes
    corr = dict(sorted(corr.items(), key=lambda item: item[0]))
    corr_disp = {}
    for key, samples in corr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        corr_disp[key] = [data[idx]['images'] for idx in idxs]
    # sort according to classes
    incorr = dict(sorted(incorr.items(), key=lambda item: item[0]))
    incorr_disp = {}
    for key, samples in incorr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        incorr_disp[key] = [data[idx]['images'] for idx in idxs]

    image_corr = draw_grid(corr_disp, n_classes, grid_size, 800, 600)
    image_incorr = draw_grid(incorr_disp, n_classes, grid_size, 800, 600)

    cv2.imshow('images correct', image_corr)
    cv2.imshow('images incorrect', image_incorr)
    cv2.waitKey()

    # this function does not return anything
    return


def display_dataset_stats(data):

    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['class']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
    # print('number of samples for each class:')
    print(class_to_num)


def balance_dataset(data, ratio):

    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data



def main():

    print('Data extraction')
    train_data = extract_data(path_temp_train, path_temp_train_i)
    display_dataset_stats(train_data)

    print('overwritting')
    train_data=balance_dataset(train_data,1.0)

    print('balancing dataset')
    display_dataset_stats(train_data)

    data_test = extract_data(path_temp_test,path_temp_test_i)
    print('test dataseet before balance')

    display_dataset_stats(data_test)

    data_test=balance_dataset(data_test,1.0)
    print('after balance test dataset:')
    display_dataset_stats(data_test)

    x=extract_data(path_temp_train,path_temp_train_i)

    data_load=new_load_data('./train')
    train_test_list()
    learn_bovw(x)

    print('extract train features')
    y=extract_features(x)
    print('training time')
    rf=train_algorithm(y)

    print('extracting test features')
    data_test=extract_features(data_test)

    print('testing dataset')
    data_test=predict(rf,data_test)

    evaluate(data_test)
    display(data_test)

    print('testing training dataset')

    train_data=predict(rf,train_data)
    evaluate(data_test)
    display(train_data)
    return


if __name__ == '__main__':
    main()



