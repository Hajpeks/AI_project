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

#IMAGE THINGS :p
from matplotlib import image
import pandas as pd
from sklearn.datasets import images

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg



# print(os.getcwd())

path_ANN = r'../train/annotations'
path_IMG=r'../train/images'

path_to_test=r'../test'
path_to_train = r'../train'




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


##tu trzeba to dodac
#list_xmls=[]
#lists_png=[]

def load_data(path):

   path_xml=os.path.abspath(os.path.join(path,'annotations/*.xml'))
   print('dlugosc xml',len(path_xml))

   path_png=os.path.abspath(os.path.join(path,'images/*.png'))
   print('dlugosc png', (len(path_png)))

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


extract_data(path_ANN)
#create_dataset(path_IMG)


data_train=load_data(path_to_train)

