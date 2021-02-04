import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH = 'Y:/CovidGAN'
training_path = os.path.join(PATH, 'Y:/CovidGAN/train')
val_path = os.path.join(PATH, 'Y:/CovidGAN/val')
testing_path = os.path.join(PATH, 'Y:/CovidGAN/test')

bs=36
img_size=112

train_datagen = ImageDataGenerator(validation_split = 0.1, shear_range=0.1, zoom_range=0.1)
test_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1)

train_data = train_datagen.flow_from_directory(training_path, subset='training', batch_size = bs, target_size = (img_size, img_size),
                                               shuffle = True, class_mode = 'binary', seed = 42)

val_data = train_datagen.flow_from_directory(training_path, subset='validation', batch_size = bs, target_size = (img_size, img_size), 
                                             shuffle = True, class_mode = 'binary', seed = 42) 

test_data = test_datagen.flow_from_directory(testing_path, batch_size = bs, target_size = (img_size, img_size),
                                             shuffle = True, class_mode = 'binary', seed = 42)

print(train_data.class_indices)

batchX, batchy = train_data.next()
print('train_batch.shape:', batchX.shape)
print('label_batch.shape', batchy.shape)
