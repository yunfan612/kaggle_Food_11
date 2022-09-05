# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:21:28 2022

@author: frank.chang
"""
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

category = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat'
            , 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

with open("./model/Transfer_InceptionResNetV2_v2.json", "r") as f:
    model = model_from_json(f.read())

model.load_weights("./model/Transfer_InceptionResNetV2_v2.hdf5")
'''
test_ds = tf.keras.preprocessing.image_dataset_from_directory('D:/food_project/test_image_v2/',
                                                          class_names = None, batch_size = 1,                                
                                                          labels = 'inferred', label_mode = 'categorical',                                                          
                                                          image_size = (256, 256), shuffle = False,                                                         
                                                          )
'''
size = (256, 256)
image_path = './data/ChineseNoodles.jpg'

try:
    image = tf.keras.preprocessing.image.load_img(image_path, target_size = size)
except Exception as e:
    #print(image_path, e)
    print('輸入錯誤','Exception',e)

input_arr = tf.keras.preprocessing.image.img_to_array(image)
#input_arr /= 255.0
input_arr = np.array([input_arr]) # Convert single image to a batch.

plt.imshow(image)   
plt.show()  
'''   
image = img_to_array(image)
image = image.astype('float32')
image /= 255.0
image = expand_dims(image, 0)
'''

#predict = np.array(model(test_ds))
predict = model.predict(input_arr)
#print(predict) # 機率list
predict_classes = np.argmax(predict, axis=1)
prediction = category[predict_classes[0]]

print(f'預測結果為: {prediction}')
print(f'機率為: {predict.ravel()[predict_classes[0]]:.2f}')

