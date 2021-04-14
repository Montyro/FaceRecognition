#Train 3 gender classifiers using only subjects from the same race (one classifier per race)
from dataloader import DiveFaceDataLoader
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import tensorflow as tf
import keras_vggface
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Activation, ActivityRegularization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, models, layers, regularizers
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from tensorflow.python.keras.backend import ndim

import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

## To solve some GPU DRAM limitations
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#load whole dataset
data = DiveFaceDataLoader().LoadData("4K_120")
#Drop Id (not required)
demo_data = data.drop('Id',axis=1)
#Set UP so its usable with keras ImageDataGenerator
demo_data.rename(columns={'Image':'filename'},inplace=True)

#Load VGGFace with resnet50 backbone
my_model = 'resnet50'
resnet = VGGFace(model = my_model)

#Select the lat leayer as feature embedding  
last_layer = resnet.get_layer('avg_pool').output
feature_layer = Flatten(name='flatten')(last_layer)
model_vgg=Model(resnet.input, feature_layer)

#Freeze the model
model_vgg.trainable = False

#Create the 3 subsets from the dataset (one per race)
#white people
white_entries = demo_data.drop(['HN','HA','MN','MA'],axis=1)
white_entries = white_entries[demo_data['HB'] != demo_data['MB']]
#asian people
asian_entries = demo_data.drop(['HN','HB','MN','MB'],axis=1)
asian_entries = asian_entries[demo_data['HA'] != demo_data['MA']]
#afroamerican people
afr_entries = demo_data.drop(['HA','HB','MB','MA'],axis=1)
afr_entries = afr_entries[demo_data['HN'] != demo_data['MN']]

#Train a classifier for gender using these datasets
white_gender_class = keras.Sequential([
    model_vgg,
    keras.layers.Dense(3000,activation="relu"),
    keras.layers.Dense(2,activation="softmax")]
)
white_gender_class.compile(loss='categorical_crossentropy',metrics=['acc'])

#Preprocess the images
def preprocess(img):
    img = np.expand_dims(img, axis=0)
    return img

#Split into training and validation
training_split  = 0.8
indxs = np.random.rand(len(white_entries)) < training_split 
training = white_entries[indxs]
training_data = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(training,directory=".",target_size=(224,224),y_col=['HB','MB'],class_mode='raw')
testing = white_entries[~indxs]
testing_data = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(testing,directory=".",target_size=(224,224),y_col=['HB','MB'],class_mode='raw')


wg_checkpoint_filepath = 'model/checkpoints/wg_c'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=wg_checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

train = True
if train:
    white_gender_class.fit(training_data,validation_data=testing_data,
                               epochs=3,
                               callbacks=[model_checkpoint_callback])









