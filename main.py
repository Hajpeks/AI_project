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


counter_rand=0

traffic_lights = []
crosswalk = []
stop = []
speedlimit = []

## translation of 4 classes to 2 classes:
##4 classes are types of road signs to 2 claasses:
#
#
# 1 - crosswalk
# 0 - not used #should be failure (other)

def xd():
    train=[]
    test=[]
    counter=0

    while len(crosswalk) != 0:
        index = random.randrange(0, len(crosswalk))
        if counter != 0:
            train.append(crosswalk[index])
        else:
            test.append(crosswalk[index])
        crosswalk.pop(index)
        counter = (counter + 1) % 4
    other=traffic_lights+speedlimit+stop
    while len(other) != 0:
        index = random.randrange(0, len(other))
        if counter != 0:
            train.append(other[index])
        else:
            test.append(other[index])
        other.pop(index)
        counter = (counter + 1) % 4
    return train,test

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
       print(len(list_xmls))
       print(len(list_png))
       for element in range(len(list_xmls)):
           data.append({'annotation':list_xmls[element],'image':cv2.imread(list_png[element]),'crosswalk':None})
   else:
       print('Data set is wrong')

   #print(data)  ## narazie 0
   return data

def get_file_list(root, file_type):
    return [os.path.join(directory_path, a) for directory_path, directory_name,
            files in os.walk(root) for a in files if a.endswith(file_type)]


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
        annotations['xmin'] = int(root.find("./object/bndbox/xmin").text)
        annotations['ymin'] = int(root.find("./object/bndbox/ymin").text)
        annotations['xmax'] = int(root.find("./object/bndbox/xmax").text)
        annotations['ymax'] = int(root.find("./object/bndbox/ymax").text)
        annotations_list.append(annotations)
    return pd.DataFrame(annotations_list)

def read_image(path):
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

data_train = data_train.reset_index()

X = data_train[['filename']]
y = data_train['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=34)


extract_data(path_ANN)
#create_dataset(path_IMG)
data_train=load_data(path_to_train)
xd()
