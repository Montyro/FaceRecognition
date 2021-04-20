#Train 3 gender classifiers using only subjects from the same race (one classifier per race)

#%% Imports
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

#%%  To solve some GPU DRAM limitations
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#%% Load dataset
#load whole dataset
demo_data = DiveFaceDataLoader().LoadData("4K_120")
#Set UP so its usable with keras ImageDataGenerator
demo_data.rename(columns={'Image':'filename'},inplace=True)

#%% Load VGGFace with resnet50 backbone
my_model = 'resnet50'
resnet = VGGFace(model = my_model)

#Select the last leayer as feature embedding  
last_layer = resnet.get_layer('avg_pool').output
feature_layer = Flatten(name='flatten')(last_layer)
model_vgg=Model(resnet.input, feature_layer)

#Freeze the model
model_vgg.trainable = False

#%% Generate subsets 
#Create the 3 subsets from the dataset (one per race)
#white people (about 55k images keep one per identity)
white_entries = demo_data.drop(['HN','HA','MN','MA'],axis=1)
white_entries = white_entries[demo_data['HB'] != demo_data['MB']]
white_entries = white_entries.drop_duplicates("Id")
#asian people
asian_entries = demo_data.drop(['HN','HB','MN','MB'],axis=1)
asian_entries = asian_entries[demo_data['HA'] != demo_data['MA']]
#afroamerican people
afr_entries = demo_data.drop(['HA','HB','MB','MA'],axis=1)
afr_entries = afr_entries[demo_data['HN'] != demo_data['MN']]
#%% Generate joint dataset
wh_mf = white_entries.rename(columns={'HB':'H','MB':'B'})
balanced_dataset_training = wh_mf[wh_mf['H'] == 1].head(250)
balanced_dataset_training = balanced_dataset_training.append(wh_mf[wh_mf['B'] == 1].head(250))
balanced_dataset_eval = wh_mf[wh_mf['H'] == 1].tail(250)
balanced_dataset_eval= balanced_dataset_eval.append(wh_mf[wh_mf['B'] == 1].tail(250)) 
as_mf = asian_entries.rename(columns={'HA':'H','MA':'B'})
balanced_dataset_training = balanced_dataset_training.append(as_mf[as_mf['H'] == 1].head(250))
balanced_dataset_training = balanced_dataset_training.append(as_mf[as_mf['B'] == 1].head(250))
balanced_dataset_eval = balanced_dataset_eval.append(as_mf[as_mf['H'] == 1].tail(250)) 
balanced_dataset_eval= balanced_dataset_eval.append(as_mf[as_mf['B'] == 1].tail(250)) 
af_mf = afr_entries.rename(columns={'HN':'H','MN':'B'})
balanced_dataset_training = balanced_dataset_training.append(af_mf[af_mf['H'] == 1].head(250))
balanced_dataset_training = balanced_dataset_training.append(af_mf[af_mf['B'] == 1].head(250))
balanced_dataset_eval = balanced_dataset_eval.append(af_mf[af_mf['H'] == 1].tail(250)) 
balanced_dataset_eval= balanced_dataset_eval.append(af_mf[af_mf['B'] == 1].tail(250)) 


#%% Create the training and testing ImageDataGenerators

#Preprocessing used for the images
def preprocess(img):
    img = np.expand_dims(img, axis=0)
    return img

#Split into training and validation
training_split  = 0.8

## White people
indxs = np.random.rand(len(white_entries)) < training_split 
training = white_entries[indxs]
print("Training entries",len(training))
training_data_white = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(training,directory=".",target_size=(224,224),y_col=['HB','MB'],class_mode='raw')
testing = white_entries[~indxs]
print("Testing entries",len(testing))
testing_data_white = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(testing,directory=".",target_size=(224,224),y_col=['HB','MB'],class_mode='raw')

## Black people
indxs = np.random.rand(len(afr_entries)) < training_split 
training = afr_entries[indxs]
print("Training entries",len(training))
training_data_black = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(training,directory=".",target_size=(224,224),y_col=['HN','MN'],class_mode='raw')
testing = afr_entries[~indxs]
print("Testing entries",len(testing))
testing_data_black= ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(testing,directory=".",target_size=(224,224),y_col=['HN','MN'],class_mode='raw')

## Asian people
indxs = np.random.rand(len(asian_entries)) < training_split 
training = asian_entries[indxs]
print("Training entries",len(training))
training_data_asian= ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(training,directory=".",target_size=(224,224),y_col=['HA','MA'],class_mode='raw')
testing = asian_entries[~indxs]
print("Testing entries",len(testing))
testing_data_asian = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(testing,directory=".",target_size=(224,224),y_col=['HA','MA'],class_mode='raw')


#%% White people  create classifier model 

white_gender_class = keras.Sequential([
    model_vgg,
    keras.layers.Dense(3000,activation="relu"),
    keras.layers.Dense(2,activation="softmax")]
)
white_gender_class.compile(loss='categorical_crossentropy',metrics=['acc'])
#%% Train a classifier for gender using these datasets
white_gender_class.fit(training_data_white,validation_data=testing_data_white,
                               epochs=3)

#%% Evaluate on other races
#Black
print("Accuracy over Black demographic group")
white_gender_class.evaluate(testing_data_black)

#Asian
print("Accuracy over Asian demographic group")
white_gender_class.evaluate(testing_data_asian)


#%% Black people create classifier model
black_gender_class = keras.Sequential([
    model_vgg,
    keras.layers.Dense(3000,activation="relu"),
    keras.layers.Dense(2,activation="softmax")]
)
black_gender_class.compile(loss='categorical_crossentropy',metrics=['acc'])
#%% Train a classifier for gender using these datasets
black_gender_class.fit(training_data_black,validation_data=testing_data_black,
                               epochs=3)

#%% Evaluate on other races
#Black
print("Accuracy over White demographic group")
black_gender_class.evaluate(testing_data_white)

#Asian
print("Accuracy over Asian demographic group")
black_gender_class.evaluate(testing_data_asian)

#%% Asian people create classifier model
asian_gender_class = keras.Sequential([
    model_vgg,
    keras.layers.Dense(3000,activation="relu"),
    keras.layers.Dense(2,activation="softmax")]
)
asian_gender_class.compile(loss='categorical_crossentropy',metrics=['acc'])
#%% Train a classifier for gender using these datasets
asian_gender_class.fit(training_data_asian,validation_data=testing_data_asian,
                               epochs=3)

#%% Evaluate on other races
#Black
print("Accuracy over Black demographic group")
asian_gender_class.evaluate(testing_data_black)

#Asian
print("Accuracy over White demographic group")
asian_gender_class.evaluate(testing_data_white)
# %%
