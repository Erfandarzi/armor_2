import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
import os

import shutil


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

path='../dataset/histopathologic-cancer-detection/train/'
annotation_file='../dataset/histopathologic-cancer-detection/train_labels.csv'
test_path='../dataset/histopathologic-cancer-detection/test/'
train_data =pd.read_csv('../dataset/histopathologic-cancer-detection/train_labels.csv')
sub = pd.read_csv('../dataset/histopathologic-cancer-detection/sample_submission.csv')

cancer = np.random.choice(train_data[train_data.label==1].id, size=13000, replace=False)
no_cancer = np.random.choice(train_data[train_data.label==0].id, size=13000, replace=False)

for m in range(2000):
        img_id = cancer[m+10000]
        shutil.copy(path + img_id + ".tif", r'..\dataset\histopathologic-cancer-detection\Testing\Cancer')

        

for n in range(2000):
        img_id = no_cancer[n+10000]
        shutil.copy(path + img_id + ".tif", r'..\dataset\histopathologic-cancer-detection\Testing\Nocancer')
     