# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:20:27 2021

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

df_train, df_valid = train_test_split(train_df, test_size = 0.3, random_state = 0)

img_h = 225
img_w = 225


train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_data_generator.flow_from_dataframe(
    dataframe = df_train, directory = 'dataset/images',
    x_col= 'id', y_col= 'label', batch_size = 16, seed = 42,
    shuffle = True, class_mode= 'categorical',
    #color_mode = 'grayscale', 
    target_size = (img_h, img_w))

validation_generator = val_data_generator.flow_from_dataframe(
    dataframe = df_valid, directory = 'dataset/images',
    x_col= 'id', y_col= 'label', batch_size = 16, seed = 42,
    shuffle = True, class_mode= 'categorical',
    #color_mode = 'grayscale',
    target_size = (img_h, img_w))

test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_data_generator.flow_from_directory(
    directory = 'dataset/images',
    shuffle = False, 
    class_mode = None,
    #color_mode = 'grayscale',
    target_size = (img_h,img_w))

test_X, test_Y = next(val_data_generator.flow_from_dataframe( 
    dataframe = df_valid, directory = 'dataset/images',
    x_col = 'id', y_col = 'label', 
    target_size = (img_h, img_w),
    batch_size = 32,
    #color_mode = 'grayscale',
    class_mode = 'categorical'
    ))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.metrics import AUC
base_model = VGG19(input_shape = (img_h, img_w, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
#x = GlobalAveragePooling2D()(base_model.output)  
x = Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.5
x = Dropout(0.5)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)

# Add a final sigmoid layer for classification
x = Dense(6, activation='softmax')(x) 

model = tf.keras.models.Model(base_model.input, x)
model.summary()
'''tf.keras.optimizers.RMSprop(lr=0.0001)'''

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['accuracy',AUC()])


from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cvs.vgg')

checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=5, min_lr=0.000001)
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=80)
callbacks_list = [checkpoint, early, reduceLROnPlat]

history = model.fit(train_generator, validation_data = validation_generator, epochs = 50, callbacks=callbacks_list)
model.save('vgg.h5')
model.load_weights(weight_path)

from tensorflow import keras
model = keras.models.load_model('vgg.h5')
pred = model.predict(test_X, batch_size = 32, verbose = 2)
results = model.evaluate(test_X, test_Y)
print(results)

df = pd.DataFrame(columns = ['actual','predicted','predicted_class'])
for i in range(len(pred)):
    df.loc[len(df.index)]=[test_Y[i],pred[i],(np.where(pred[i] == np.max(pred[i]))[0])+1]

df.to_csv('actual vs pred.csv')



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

'''
fig, axc = plt.subplots(figsize = (7,7))
axc.plot(history.history['auc_1'])
axc.plot(history.history['val_auc_1'])
axc.legend(['train', 'test'], loc='upper left')
axc.set_xlabel('epoch')
axc.set_ylabel('auc')
fig.savefig('auc.png', dpi = 300)


fig, axx = plt.subplots(figsize = (7,7))
axx.plot(history.history['mse'])
axx.plot(history.history['val_mse'])
axx.legend(['train', 'test'], loc='upper left')
axx.set_xlabel('epoch')
axx.set_ylabel('mse')
fig.savefig('mse.png', dpi = 300)
'''
