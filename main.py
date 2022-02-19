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

#IMAGE THINGS :p
from matplotlib import image
import pandas as pd
from sklearn.datasets import images

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg



# print(os.getcwd())

path = r'../train/annotations'
path_to_image = r'../train'



## translation of 4 classes to 2 classes:
##4 classes are types of road signs to 2 claasses:
#
#
# 1 - crosswalk
# 0 - not used #should be failure (other)


class_name = {'speedlimit': 0,
              'crosswalk': 1,
              'trafficlight': 0,
              'stop': 0}

def extract_data(path):
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
    traffic_lights = []
    crosswalk = []
    stop = []
    speedlimit = []

    allxml=[]

    for entry in os.scandir(path):
        tree = ET.parse(entry)
        root = tree.getroot()
        # name=root.findall('object')
        for filename in root.findall('object'):
            name = filename.find('name').text
            #print(name)
            if name == 'trafficlight':
                traffic_lights.append(name)
            elif name == 'stop':
                stop.append(name)
            elif name == 'speedlimit':
                speedlimit.append(name)
            elif name == 'crosswalk':
                crosswalk.append(name)
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

def load_data(path, filename):
    """
    Loads data from disk.
    @param path: Path to dataset directory.
    @param filename: Filename of csv file with information about samples.
    @return: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    """
   # entry_list = pandas.read_csv(os.path.join(path, filename))
   #
   #  data = []
   #  for idx, entry in entry_list.iterrows():  # co to iterowrs xd
   #      class_id = class_id_to_new_class_id[entry['ClassId']]   #simplified class
   #      image_path = entry['Path']  #image path
   #
   #      if class_id != -1: #if not -1 class(as we can describe it)
   #          image = cv2.imread(os.path.join(path, image_path)) #load image from specified file
   #          #if failed returns empty matrix
   #          data.append({'image': image, 'label': class_id})
   #
   #  return data

######### Whole path import






# plt.figure(figsize=(30, 30))
# for i in range(1):
#     file = random.choice(os.listdir(path_to_image))
#     image_path = os.path.join(path_to_image, file)
#     img = mpimg.imread(image_path)
#     ax = plt.subplot(1, 5, i + 1)
#     ax.title.set_text(file)
#    # plt.imshow(img)
#    # plt.show()



def create_dataset(path_to_image):
    img_data_array=[]
    class_name=[]
    IMG_HEIGHT = 100
    IMG_WIDTH = 100
    counter=0
    for dir1 in os.listdir(path_to_image):
        for file in os.listdir(os.path.join(path_to_image, dir1)):

            image_path= os.path.join(path_to_image, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB) #cv2.color... to RGB
            if (type(image) == type(None)):
                pass
                print('failuer')
            else:
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
                counter+=1
                #print(img_data_array)
                # print(class_name)
        print(counter)
    return img_data_array, class_name



extract_data(path)
create_dataset(path_to_image)




