import os
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH = 'Y:/CovidGAN'

# MIXED_IMAGES
# CORONA - 280
# NORMAL - 480
TESTING_PATH = os.path.join(PATH, 'Y:/CovidGAN/TESTING_DATA(original_images_only)')

files1 = os.listdir("Y:/CovidGAN/TESTING_DATA(original_images_only)/CORONA")
train_corona = len(files1)
files2 = os.listdir("Y:/CovidGAN/TESTING_DATA(original_images_only)/NORMAL")
train_normal = len(files2)

print("Test corona: ", train_corona)
print("Test normal: ", train_normal)

train_datagen = ImageDataGenerator(validation_split = 0.1)

train_data = train_datagen.flow_from_directory(TESTING_PATH, subset = 'training', target_size = (112, 112),
                                               class_mode = 'categorical', batch_size = 16, shuffle = True)
test_data = train_datagen.flow_from_directory(TESTING_PATH, subset = 'validation', target_size = (112, 112),
                                               class_mode = 'categorical', batch_size = 16, shuffle = True)

print(train_data.class_indices)

from keras.applications.vgg19 import VGG19

# define cnn model
def define_model():
    # load model
    model = VGG19(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    avg_pool = GlobalAveragePooling2D()(model.output)
    dense1 = Dense(62, activation='relu')(avg_pool)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(2, activation='softmax')(dropout)    
    
    # define new model
    model = Model(inputs = model.inputs, outputs = dense2)
    # compile model
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    model.fit(train_data, validation_data=test_data, validation_steps=len(test_data), steps_per_epoch=len(train_data), epochs=20, verbose=1)

    return model
    
model = define_model()
