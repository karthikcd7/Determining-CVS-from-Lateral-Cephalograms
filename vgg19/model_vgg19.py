# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:47:32 2022

@author: Karthik
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import datetime, os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

train_df = pd.read_csv('dataset/csv/dataset.csv')
print(train_df.head())

df_train, df_valid = train_test_split(train_df, test_size = 0.20, random_state = 0)

img_h = 225
img_w = 225


#train_data_generator = ImageDataGenerator(rescale=1.0/255.0)
val_data_generator = ImageDataGenerator(rescale=1./255)

train_data_generator = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)
image_dir = 'dataset/images/bg double'

train_generator = train_data_generator.flow_from_dataframe(
    dataframe = df_train, directory = image_dir,
    x_col= 'id', y_col= 'label', batch_size = 16,
    shuffle = True, class_mode= 'categorical', 
    target_size = (img_h, img_w))



validation_generator = val_data_generator.flow_from_dataframe(
    dataframe = df_valid, directory = image_dir,
    x_col= 'id', y_col= 'label', batch_size = 16,
    shuffle = True, class_mode= 'categorical',  
    target_size = (img_h, img_w))


test_X, test_Y = next(val_data_generator.flow_from_dataframe( 
    dataframe = df_valid, directory =  image_dir,
    x_col = 'id', y_col = 'label', 
    target_size = (img_h, img_w),
    batch_size = 16,
    class_mode = 'categorical'))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import SGD

base_model = VGG19(input_shape = (img_h, img_w, 3), include_top = False, weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(6, activation='softmax')(x) 

model = tf.keras.models.Model(base_model.input, x)
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001) , loss = 'categorical_crossentropy',metrics = ['accuracy',AUC()])



from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cvs.vgg')

checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=30, verbose=1, mode='auto', epsilon=0.001, cooldown=5, min_lr=0.000001)
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=80)
callbacks_list = [checkpoint, early, reduceLROnPlat]

history = model.fit(train_generator, validation_data = validation_generator, epochs = 100, callbacks=callbacks_list)

model.load_weights(weight_path)

from tensorflow import keras
model.save('vgg.h5')
model = keras.models.load_model('vgg.h5')
pred = model.predict(test_X, batch_size = 16, verbose = 2)
results = model.evaluate(test_X, test_Y)
print(results)



fig, axss = plt.subplots(figsize = (7,7))
axss.plot(history.history['loss'])
axss.plot(history.history['val_loss'])
axss.legend(['train', 'test'], loc='upper left')
axss.set_xlabel('epoch')
axss.set_ylabel('loss')
fig.savefig('loss.png', dpi = 300)


fig, axs = plt.subplots(figsize = (7,7))
axs.plot(history.history['accuracy'])
axs.plot(history.history['val_accuracy'])
axs.legend(['train', 'test'], loc='upper left')
axs.set_xlabel('epoch')
axs.set_ylabel('accuracy')
fig.savefig('accuracy.png', dpi = 300)

