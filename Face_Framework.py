# -*- coding: utf-8 -*-

#%%
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
#from tensorflow.python.keras.preprocessing.dataset_utils import labels_to_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf

from tensorflow.compat.v1 import InteractiveSession

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#%%
#Import the ResNet-50 model trained with VGG2 database
my_model = 'resnet50'
resnet = VGGFace(model = my_model)
#resnet.summary()  

#Select the lat leayer as feature embedding  
last_layer = resnet.get_layer('avg_pool').output
feature_layer = Flatten(name='flatten')(last_layer)
model_vgg=Model(resnet.input, feature_layer)
 


# TASK 1: Read the DiveFace database and obtain the embeddings of 50 face images (1 image per subject) from 
# the 6 demographic groups (50*6=300 embeddings in total).

# Read the dataset

# Link to the database: https://dauam-my.sharepoint.com/:u:/g/personal/aythami_morales_uam_es/ERd0YZG26FlGl1hr9nQtd54BNmW2XMwuzS-LXh0DoMp2ig?e=f8jD7w

#%%
from dataloader import DiveFaceDataLoader


data = DiveFaceDataLoader().LoadData("4K_120")
#%%

from PIL import Image
#Get 50 from each demographic group, max 1 example per ID.
#set id as index
one_id = data.drop_duplicates("Id")
data_id_as_index = data.set_index("Id",drop=True)
print(one_id.columns.array[1:])
embeddings = []

print(one_id)
num_persons = 1200
for column in one_id.columns.array[1:]:
    #print(column)
    top50 =  one_id.loc[one_id[column]==1].head(num_persons)
    for index,row in top50.iterrows():
        #print(row)
        img = Image.open(row[0])
        img = img.resize((224,224))
        img = np.expand_dims(img, axis=0)
        embed = model_vgg.predict(img)
        
        embeddings.append(embed[0])

#print(embeddings)
#TASK 2: Using t-SNE, represent the embeddings and its demographic group. Can you differenciate the different demographic groups?
#%%

from sklearn.manifold import TSNE
labels =  [i//num_persons for i in np.arange(6*num_persons)]
tsne = TSNE(2,random_state=0)
tsne_data = tsne.fit_transform(embeddings)
print(len(labels))
print(tsne_data.shape)
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(tsne_data[:-1,0],tsne_data[:-1,1], labels, c = labels, cmap = 'tab10')
handles, labels = scatter.legend_elements()
legend = ax.legend(handles = handles, labels = labels)
plt.title('Demographic')
plt.show()



#%%
# TASK 3: Using the ResNet-50 embedding (freeze the model), train your own attribute classifiers (ethnicity and gender). 
# Recommendation: use a simple dense layer with a softmax output. Divide DiveFace into train and test.
model_vgg.trainable = False

demographic_classification = keras.Sequential([
    model_vgg,
    keras.layers.Dense(3000,activation="relu"),
    keras.layers.Dense(6,activation="softmax")]
)

demographic_classification.compile(loss='categorical_crossentropy',metrics=['acc'])

#%%
## Divide dataset into training and validation
#As the dataset is balanced, we randomly select rows for  training and test

training_split = 0.8
#we dont need id for demographic group classification

demo_data = data.drop('Id',axis=1)
demo_data.rename(columns={'Image':'filename'},inplace=True)
print(demo_data)
#demo_data = demo_data.reset_index()
indxs = np.random.rand(len(demo_data)) < training_split 

training  = demo_data[indxs]

def preprocess(img):
    img = np.expand_dims(img, axis=0)
    return img

training_data = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(training,directory=".",target_size=(224,224),y_col=['HA','HB','HN','MA','MB','MN'],class_mode='raw')
#training_data = tf.data.Dataset.from_tensor_slices((training.index.array,training.values))

testing = demo_data[~indxs]
testing_data = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(testing,directory=".",target_size=(224,224),y_col=['HA','HB','HN','MA','MB','MN'],class_mode='raw')

checkpoint_filepath = 'model/checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

train = False
if train:
    demographic_classification.fit(training_data,validation_data=testing_data,
                               epochs=3,
                               callbacks=[model_checkpoint_callback])
else:
    demographic_classification.load_weights(checkpoint_filepath)


## TSNE trained model
from sklearn.manifold import TSNE

#get all labels:
indxs = np.random.rand(len(demo_data)) < 0.05 

whole_dataset = ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(demo_data,directory=".",target_size=(224,224),y_col=['HA','HB','HN','MA','MB','MN'],class_mode='raw')
labels = demographic_classification.predict(whole_dataset)

print("labels")
tsne = TSNE(2,random_state=0)
tsne_data = tsne.fit_transform(embeddings)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(tsne_data[:,0],tsne_data[:,1], labels, c = labels, cmap = 'tab10')
handles, labels = scatter.legend_elements()
legend = ax.legend(handles = handles, labels = labels)
plt.title('Demographic')
plt.show()

# %%


