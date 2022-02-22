"""Aditional definitions/import"""
#from keras.utils import to_categorial
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

import os
import random
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



# print(os.getcwd())

path_ANN = r'../archiwum/annotations'
path_IMG=r'../archiwum/images'

path_temp_test=r'../test/annotations'
path_temp_train= r'../train/annotations'

path_temp_test_i=r'../test/images'
path_temp_train_i= r'../train/images'

path_to_test=r'../test'
path_to_train = r'../train'


counter_rand=0

#4 classes
crosswalk = []
other=[]

#train and test
train = []
test = []



def old_extract_data(path):
    """Extracting needed data from .xml file"""
            #LIGHT THOUGHTS
    #print(os.getcwd())
    #l = list(os.listdir())
    #print(l)

    """NEW LISTS AND COUNTER+TOTAL CHECK"""
    total = 0
    cnt = 0
    counterofxmls=0
    ###empty lists
    # traffic_lights = []
    # crosswalk = []
    # stop = []
    # speedlimit = []

    allxml=[]

    for entry in os.scandir(path):
        tree = ET.parse(entry)
        root = tree.getroot()
        # name=root.findall('object')
        for filename in root.findall('object'):
            name = filename.find('name').text
            #print(name)
            if name == 'trafficlight':
                traffic_lights.append(entry)
            elif name == 'stop':
                stop.append(entry)
            elif name == 'speedlimit':
                speedlimit.append(entry)
            elif name == 'crosswalk':
                crosswalk.append(entry)
            cnt += 1
        # print(name)
        #         cnt += 1
        if entry.is_dir(follow_symlinks=False):
            total += extract_data(entry.path)
        else:
            total += entry.stat(follow_symlinks=False).st_size

        for nameroad in root.findall('annotations'):
            filnm=nameroad.find('filename').text
            allxml.append(filnm)
            counterofxmls+=1
            #print(filnm)
    # print(total)
    # print(cnt)
    #print(counterofxmls)
    #print(allxml)

    #####PRINTS OF ALL TYPES (HMM CLASSES)

    print('traffic_light', len(traffic_lights))
    print('stop', len(stop))
    print('speedlimit', len(speedlimit))
    print('crosswalk', len(crosswalk))
    print('filename', len(allxml))

    return total

def load_data(path):

   path_xml=os.path.abspath(os.path.join(path,'annotations/*.xml'))
   #print('dlugosc xml',len(path_xml))

   path_png=os.path.abspath(os.path.join(path,'images/*.png'))
   #print('dlugosc png', (len(path_png)))

   list_xmls = glob.glob(path_xml)
   list_png = glob.glob(path_png)

   list_xmls.sort()
   list_png.sort()

   data=[] #jeden kontener na dane

   if(len(list_xmls)==len(list_png)):
       #print(len(list_xmls))
       #print(len(list_png))
       for element in range(len(list_xmls)):
           data.append({'annotation':list_xmls[element],'image':cv2.imread(list_png[element]),'crosswalk':None})
   else:
       print('Data set is wrong')

   #print(data)  ## narazie 0
   return data

def get_file_list(root, file_type):
    return [os.path.join(directory_path, a) for directory_path, directory_name,
            files in os.walk(root) for a in files if a.endswith(file_type)] ##zczytuje kazdy folder


def load_data_2(path_annotations):
    annotations_path_list = get_file_list(path_annotations, '.xml')
    for a_path in annotations_path_list:
        root = ET.parse(a_path).getroot()
        if root.find("./object/name").text == 'crosswalk':
            crosswalk.append(a_path)
        else:
            other.append(a_path)

def extract_data(path_annotations, path_images):
    """Extracting needed data from .xml file and fitting to image"""
    annotations_path_list = get_file_list(path_annotations, '.xml')
    annotations_list = []
    for a_path in annotations_path_list:
        root = ET.parse(a_path).getroot()
        annotations = {}
        annotations['filename'] = Path(str(path_images) + '/' + root.find("./filename").text)
        annotations['width'] = root.find("./size/width").text
        annotations['height'] = root.find("./size/height").text
        annotations['class'] = root.find("./object/name").text
        #boundbox description
        annotations['xmin'] = int(root.find("./object/bndbox/xmin").text)
        annotations['ymin'] = int(root.find("./object/bndbox/ymin").text)
        annotations['xmax'] = int(root.find("./object/bndbox/xmax").text)
        annotations['ymax'] = int(root.find("./object/bndbox/ymax").text)
        annotations_list.append(annotations)
    return pd.DataFrame(annotations_list)


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
    ## annotations
    for clear in os.listdir(path_temp_train):
        os.unlink(os.path.join(path_temp_train,clear))
    for clear in os.listdir(path_temp_test):
        os.unlink(os.path.join(path_temp_test, clear))
     ## images
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


# def read_image(path):
#     image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
#     scale_percent=60
#     width=int(image.shape[1]* scale_percent/100)
#     height=int(image.shape[0]*scale_percent/100)
#     dim=(width,height)
#
#     resize=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
#
#     print('Resized:',resize.shape)


#data_train=extract_data(path_ANN,path_IMG)

#old_extract_data(path_ANN)
load_data_2(path_ANN)
train_test_list()

x=extract_data(path_temp_train)  ## najwazniejsza (x wszystko przeiterowac po x)

#read_image(path_IMG)


print('crosswalk', len(crosswalk))
print('other',len(other))

print('train_list',len(train))
print('test_list',len(test))
