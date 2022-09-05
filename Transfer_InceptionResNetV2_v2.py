# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:23:49 2022

@author: frank.chang
"""
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.applications import InceptionResNetV2
import h5py
from sklearn.utils import class_weight
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.utils import to_categorical
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
import itertools

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

K.clear_session()

category = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat'
            , 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

train_ds = tf.keras.preprocessing.image_dataset_from_directory('D:/database/Kaggle_Food_11_image_dataset/training/',
                                                          class_names = category, batch_size = 32,                                
                                                          labels = 'inferred', label_mode = 'categorical',                                                          
                                                          image_size = (256, 256), shuffle = True,                                                         
                                                          )

validation_ds = tf.keras.preprocessing.image_dataset_from_directory('D:/database/Kaggle_Food_11_image_dataset/validation/',
                                                          class_names = category, batch_size = 32,                                
                                                          labels = 'inferred', label_mode = 'categorical',                                                          
                                                          image_size = (256, 256), shuffle = True,                                                         
                                                          )

test_ds = tf.keras.preprocessing.image_dataset_from_directory('D:/database/Kaggle_Food_11_image_dataset/evaluation/',
                                                          class_names = category, batch_size = 32,                                
                                                          labels = 'inferred', label_mode = 'categorical',                                                          
                                                          image_size = (256, 256), shuffle = False,                                                         
                                                          )

train_ds = train_ds.prefetch(buffer_size=32)  #在訓練時，同時讀取下一批資料，並作轉換
validation_ds = validation_ds.prefetch(buffer_size=32)

#------------計算class weight-------------
train_label = []
train_numb = len(np.concatenate([i for x, i in train_ds], axis=0))
for images, labels in train_ds.unbatch().take(train_numb):
    train_label.append(labels.numpy())
train_label = np.array(train_label)
train_label = np.argmax(train_label, axis=1)

cw = class_weight.compute_class_weight('balanced', classes = np.unique(train_label), y = train_label)
cw_dict = dict(enumerate(cw))
#---------------------------------------

model = Sequential()
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 3)))
model.add(InceptionResNetV2(input_shape=(256, 256, 3), include_top=False))
model.add(GlobalAveragePooling2D())
model.add(Dense(11, activation='softmax'))

print(model.summary()) # show model summary
plot_model(model, show_shapes=True, to_file= './Transfer_InceptionResNetV2.png') # print model graph

adamax = Adamax(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="./model/Transfer_InceptionResNetV2_v2.hdf5", verbose=1, save_best_only=True, monitor="val_loss", mode="min")

with open("./model/Transfer_InceptionResNetV2_v2.json", "w") as f:
    f.write(model.to_json())

#history = model.fit(train_ds, validation_data = validation_ds, class_weight = cw_dict, epochs=20, batch_size=8, verbose=2, callbacks=[checkpointer], shuffle=True)
history = model.fit(train_ds, validation_data = validation_ds, epochs=10, verbose=2, callbacks=[checkpointer])

#---------------繼續訓練------------
'''
with open("D:/food_project/model/Transfer_InceptionResNetV2_v2.json", "r") as f:
    model = model_from_json(f.read())

model.load_weights("D:/food_project/model/Transfer_InceptionResNetV2_v2.hdf5")

#adamax = Adamax(lr=1e-4)
#model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])


model.trainable = True  #想要讓某層參加訓練，必須'先'讓全部層[可訓練]，'再'讓不想參加訓練的層[凍結].

for layer in model.layers[:-2]:
    layer.trainable = False

adam = Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="D:/food_project/model/Transfer_InceptionResNetV2_v2.hdf5", verbose=1, save_best_only=True, monitor="val_loss", mode="min")
history = model.fit(train_ds, validation_data = validation_ds, class_weight = cw_dict, epochs=20, batch_size=16, verbose=2, callbacks=[checkpointer], shuffle=True)
'''
#------------------------------------

plt.plot(history.history['accuracy'])    
plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(test_ds)
y_pred = model.predict(test_ds, batch_size=32)
#predict_classes = model.predict_classes(data_evaluation)

y_true = []
test_numb = len(np.concatenate([i for x, i in train_ds], axis=0))
for images, labels in test_ds.unbatch().take(test_numb):
    y_true.append(labels.numpy())
y_true = np.array(y_true)

y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)
print(cm)
plot_confusion_matrix(cm, category, normalize=False, title='Confusion Matrix')

print(f'The accuracy of the model is {accuracy:.2f}\nThe Loss in the model is {loss:.2f}')

class_numb = np.size(cm,axis = 1)
for i in range(class_numb):
    acc = cm[i, i] / sum(cm[i, :])
    print (f'{category[i]} accuracy : {acc:.2f}')
   
print(classification_report(y_true, y_pred, target_names = category))

