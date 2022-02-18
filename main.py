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

import xml.etree.ElementTree as ET


# translation of 43 classes to 3 classes:
# 0 - prohibitory
# 1 - warning
# 2 - mandatory
# -1 - not used
# class_id_to_new_class_id = {0: 0,
#                             1: 0,
#                             2: 0,
#                             3: 0,
#                             4: 0,
#                             5: 0,
#                             6: -1,
#                             7: 0,
#                             8: 0,
#                             9: 0,
#                             10: 0,
#                             11: 1,
#                             12: -1,
#                             13: 1,
#                             14: 0,
#                             15: 0,
#                             16: 0,
#                             17: 0,
#                             18: 1,
#                             19: 1,
#                             20: 1,
#                             21: 1,
#                             22: 1,
#                             23: 1,
#                             24: 1,
#                             25: 1,
#                             26: 1,
#                             27: 1,
#                             28: 1,
#                             29: 1,
#                             30: 1,
#                             31: 1,
#                             32: -1,
#                             33: 2,
#                             34: 2,
#                             35: 2,
#                             36: 2,
#                             37: 2,
#                             38: 2,
#                             39: 2,
#                             40: 2,
#                             41: -1,
#                             42: -1}
#

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


#load_mine()

patha = r'C:\Users\hpiet\PycharmProjects\WhereIsTheCrosswalk\train\annotations\road0.xml'
pathi= r'C:\Users\hpiet\PycharmProjects\WhereIsTheCrosswalk\train\images\road0.png'
    # implementacja ET

tree=ET.parse(r'train\annotations\road0.xml')
root=tree.getroot()

for filename in root.findall('object'):
    name=filename.find('name').text
    print(name)

#root.findall("*/filename")
# x=root.tag
# y=root.attrib
# print(x,y)
# for child in root:
#     print(child.tag,child.attrib)

    # Using cv2.imread() method
  #  img = cv2.imread(path)

    # Displaying the image
   # cv2.imshow('image', img)

# def main():
#     data_train = load_data('./', 'Train.csv')
#     print('train dataset before balancing:')
#     display_dataset_stats(data_train)
#     data_train = balance_dataset(data_train, 1.0)
#     print('train dataset after balancing:')
#     display_dataset_stats(data_train)
#
#     data_test = load_data('./', 'Test.csv')
#     print('test dataset before balancing:')
#     display_dataset_stats(data_test)
#     data_test = balance_dataset(data_test, 1.0)
#     print('test dataset after balancing:')
#     display_dataset_stats(data_test)
#
#     # you can comment those lines after dictionary is learned and saved to disk.
#  #   print('learning BoVW')
#  #   learn_bovw(data_train)
#
#     print('extracting train features')
#     data_train = extract_features(data_train)
#
#     print('training')
#     rf = train(data_train)
#
#     print('extracting test features')
#     data_test = extract_features(data_test)
#
#     print('testing on testing dataset')
#     data_test = predict(rf, data_test)
#     evaluate(data_test)
#     display(data_test)
#
#     print('testing on trening dataset')
#     data_train = predict(rf, data_train)
#     evaluate(data_test)
#     display(data_train)
#
#     return#
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
