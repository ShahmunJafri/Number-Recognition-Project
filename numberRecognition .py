#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
import glob
import random

class DataGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        
    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (28,28))
               for file_name in batch_x]), np.array(batch_y)


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten



image_path = '/Users/shahmun/Downloads/1272_2280_bundle_archive/trainingSet/trainingSet'
train_idx = []
val_idx = [] 
main_labels = [] 
train_labels = []
val_labels = []


num_epochs = 100

main_labels = glob.glob(image_path +'/*/*.jpg') #putting all file name in "main_labels"


random.shuffle(main_labels) #randomizing order

for x in range(0, int(((len(main_labels)-1)*9)/10)):
    train_labels.append(main_labels[x]) 
    
for x in range(int(((len(main_labels) - 1)*9)/10), len(main_labels)): #@tawe141
    val_labels.append(main_labels[x])

for x in range(0, int(((len(main_labels) - 1)*9)/10)): #9/10 because I want 1/10 of the training set to go to the validation set
    train_idx.append((main_labels[x].strip(image_path)).strip('.jpg'))#im stripping the image path and .jpg to just get the
                                                                 #label of each image 
                            
for x in range(int(((len(main_labels) - 1)*9)/10), len(main_labels)): 
     val_idx.append((main_labels[x].strip(image_path)).strip('.jpg')) #1/10 of training set going to validation set
    
training_generator = DataGenerator(train_labels, train_idx, 64)
validation_generator = DataGenerator(val_labels, val_idx, 64)


#create model #use 

#model = Sequential()
# model.fit_generator(generator=training_generator,
#                                         steps_per_epoch=(64),
#                                          epochs=num_epochs,
#                                          verbose=1,
#                                          validation_data=validation_generator,
#                                         validation_steps=(64),
#                                          use_multiprocessing=True,
#                                          workers=2,
#                                    max_queue_size=32)
                              
# Design model

#[...] # Architecture
#model.compile()

# Train model on dataset
#model.fit(training_generator, validation_data=validation_generator)



# In[3]:


# training_generator[0]
# print(training_generator[0]) 
assert len(training_generator[0]) == 2 
assert training_generator[0][0].shape == (64, 28, 28) 


# In[4]:


print(main_labels)


# In[ ]:




