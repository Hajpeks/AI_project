# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# !/usr/bin/env python

"""code template"""

import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
# from sklearn.esemble import confusion_matrix
from sklearn.metrics import confusion_matrix
#from keras.utils import to_categorial
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


import xml.etree.ElementTree as ET


# translation of 43 classes to 3 classes:
# 0 - prohibitory
# 1 - warning
# 2 - mandatory
# -1 - not used

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

def load_mine():
    traffic_lights=[]
    crosswalk=[]
    stop=[]
    speedlimit=[]
    patha = r'C:\Users\hpiet\PycharmProjects\WhereIsTheCrosswalk\train\annotations\road0.xml'
    pathi= r'C:\Users\hpiet\PycharmProjects\WhereIsTheCrosswalk\train\images\road0.png'
    # implementacja ET

    root=ET.fromstring(patha)
    root.findall(object)
    return object


######### Whole path import

#filedirectory="train/annotations/"
#directory=os.fsencode(filedirectory)

#road="road0"

#for file in os.listdir(directory):
#    filename=os.fsdecode(road)
#    if filename.endswith(".xml"):
#        continue

#load_mine()

#path = r'C:\Users\hpiet\PycharmProjects\WhereIsTheCrosswalk\train\annotations\road0.xml'
#pathi= r'C:\Users\hpiet\PycharmProjects\WhereIsTheCrosswalk\train\images\road0.png'
    # implementacja ET

            ##ELEMENT TREE


# tree=ET.parse(r'train\annotations\road0.xml')
# root=tree.getroot()

# for filename in root.findall('object'):
#     name=filename.find('name').text
#     print(name)

# dlugosc=877
# nazwa='road'
# for i in range(dlugosc):
#     i=str.=nazwa+'i'
#     print(nazwa)
print('******')
print(os.getcwd())
l = list(os.listdir())
print(l)
path = r'./train/annotations'


######## counter and path to folder

path = r'./train/annotations'


def get_tree_size(path):
    """Return total size of files in given path and subdirs."""
    total = 0
    cnt = 0
###empty lists
    traffic_lights = []
    crosswalk = []
    stop = []
    speedlimit = []

    for entry in os.scandir(path):
        tree = ET.parse(entry)
        root = tree.getroot()
        #name=root.findall('object')
        for filename in root.findall('object'):
            name=filename.find('name').text
            print(name)
            if name=='trafficlight':
                traffic_lights.append(name)
            elif name=='stop':
                stop.append(name)
            elif name == 'speedlimit':
                speedlimit.append(name)
            elif name=='crosswalk':
                crosswalk.append(name)
            cnt+=1
        #print(name)
        #         cnt += 1
        if entry.is_dir(follow_symlinks=False):
            total += get_tree_size(entry.path)
        else:
            total += entry.stat(follow_symlinks=False).st_size
    #print(total)
    #print(cnt)
    print('traffic_light',len(traffic_lights))
    print('stop',len(stop))
    print('speedlimit',len(speedlimit))
    print('crosswalk',len(crosswalk))
    return total



# tree = ET.parse(r'train\annotations\road0.xml')
# root = tree.getroot()
#with os.scandir(os.path.join(path)) as it:
  #  print(os.getcwd())
#     for element in path:
#         tree = ET.parse(element)
#         root = tree.getroot()
#         name=root.findall('object')
#         print(name)
#         cnt += 1
#     print(cnt)


get_tree_size(path)






# root.findall("*/filename")
# x=root.tag
# y=root.attrib
# print(x,y)


# for child in root:
#     print(child.tag,child.attrib)

    # Using cv2.imread() method
  #  img = cv2.imread(path)

    # Displaying the image
   # cv2.imshow('image', img)

