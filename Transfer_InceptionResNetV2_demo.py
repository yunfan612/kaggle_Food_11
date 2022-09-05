# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:08:19 2022

@author: frank.chang
"""

import tensorflow as tf
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.models import model_from_json
import numpy as np
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

warnings.filterwarnings('ignore')
path = os.getcwd()

category = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat'
            , 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

size = (256, 256)

with open(path + '/model/Transfer_InceptionResNetV2_v2.json', "r") as f:
    model = model_from_json(f.read())
        
model.load_weights(path + '/model/Transfer_InceptionResNetV2_v2.hdf5')

idx = 0

while idx == 0:
    input_data_path = input("input data file：")
    input_data_path = input_data_path.replace('\\', '/')
     
    try:
        image = load_img(input_data_path, target_size = size)
    except Exception as e:
        #print(image_path, e)
        print('輸入錯誤','Exception',e)
        
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    
    predict = np.array(model(image))
    #print(predict) # 機率list
    predict_classes = np.argmax(predict, axis=1)
    prediction = category[predict_classes[0]]

    print(f'預測結果為: {prediction}')
    print(f'機率為: {predict.ravel()[predict_classes[0]]:.2f}')