#Dataloader for DiveFace dataset

#Structure:
#4K_120
#| gr4K_120
#| |entry
#| | | photo 1
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import pandas as pd

class DiveFaceDataLoader():
    
    def __init__(self):
        self.image_size = 224

    def LoadData(self, route):
        total_entries = ([],[])
        class_folders = os.listdir(route)
        #df = pd.DataFrame(columns=["Image","Id","Gender","Race"])
        data = []
        id_counter = 0
        #format is going to be entry | id | gender | race
        for category in class_folders:
            gender= category[0]
            race = category[1]

            for id in os.listdir(route+"/"+category):
                for image in os.listdir(route+"/"+category+"/"+id):
                    image_route = route+"/"+category+"/"+id+"/"+image
                    
                    #add entry to dataset
                    #img = Image.open(image_route)
                    entry = [id_counter,(0 if gender == 'H' else 1),(0 if race == 'A' else ( 1 if race == "B" else 2))]
                    total_entries[0].append(image_route)
                    total_entries[1].append(entry)
                    #serie = {"Image":image_route,"Id":id_counter,"Gender":(0 if gender == 'H' else 1),"Race":(0 if race == 'A' else ( 1 if race == "B" else 2))}
                    #onehot
                    serie = {"Image":image_route,
                             "Id": id_counter,
                             "HA":1 if entry[1]== 0 and entry[2]== 0 else 0,
                             "HB":1 if entry[1]== 0 and entry[2]== 1 else 0,
                             "HN":1 if entry[1]== 0 and entry[2]== 2 else 0,
                             "MA":1 if entry[1]== 1 and entry[2]== 0 else 0,
                             "MB":1 if entry[1]== 1 and entry[2]== 1 else 0,
                             "MN":1 if entry[1]== 1 and entry[2]== 2 else 0}

                    data.append(serie)
                    
                id_counter += 1
        
        
#       dataset = tf.data.Dataset.from_tensor_slices((total_entries[0], total_entries[1]))
        df = pd.DataFrame(data)
        #df = df.set_index("Image",drop=True)
        #df.append(data)
        return df

        





