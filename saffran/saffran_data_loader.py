import numpy as np
from os import path
import os 
import pandas as pd


def load_saffran_dataset(dataroot='./data/Saffron_Dataset/Labeled/'):
    img_paths, bboxes = [], []
    files = os.listdir(dataroot)
    jpegFiles = [i for i in files if i.endswith('.jpg')]

    for image_name in jpegFiles:
        name = image_name.split('.')[0]
        csv_name = 'box_{}.csv'.format(name)
        box = np.array(pd.read_csv(dataroot+csv_name))
        img_paths.append(path.join(dataroot+image_name))
        bboxes.append(box)
    return np.array(img_paths), np.array(bboxes)
    
