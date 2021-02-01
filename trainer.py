import os 
import cv2
import pickle
import numpy as np
import collections
from collections import defaultdict
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

nb_epochs = 1200
batch_size = 36
latent_dim = 100

train_history = defaultdict(list)
test_history = defaultdict(list)
       
combined, dis, gen = define_gan(latent_dim = 100)

for epoch in range(nb_epochs):
    print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

    nb_batches = int(1332/batch_size)
    progress_bar = Progbar(target=nb_batches)

    epoch_gen_loss = []
    epoch_disc_loss = []
    gen_test_loss = []
    disc_test_loss = []

    for index, (image_batch, label_batch) in zip(range(nb_batches), train_data):

        image_batch = image_batch * (1. / 127.5) - 1

        progress_bar.update(index)

        noise, sampled_labels = generate_batch_noise_and_labels(batch_size, latent_dim)
        
        generated_images = gen.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)
        
        X = np.concatenate((image_batch, generated_images))
        
        valid = label_smoothing(vector = valid_o, max_dev = 0.2)
        fake = label_smoothing(vector = fake_o, max_dev = 0.2)
        
        y = np.concatenate((valid, fake), axis = 0)
        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
        
        epoch_disc_loss.append(dis.train_on_batch(X, [y, aux_y]))

        noise, sampled_labels = generate_batch_noise_and_labels(2 * batch_size, latent_dim)
                
        trick = np.ones(2 * batch_size)

        epoch_gen_loss.append(combined.train_on_batch(
            [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))
        
        
    for test_image_batch, test_label_batch in test_data:
                
        print('\nTesting for epoch {}:'.format(epoch + 1))

        test_image_batch = test_image_batch * (1. / 127.5) - 1

        nb_train = test_image_batch.shape[0]
        #nb_test = test_label_batch

        noise, sampled_labels = generate_batch_noise_and_labels(nb_train, latent_dim)

        generated_images = gen.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False
        )  
            
        X = np.concatenate((test_image_batch, generated_images))
        y = np.array([1] * nb_train + [0] * nb_train)
        aux_y = np.concatenate((test_label_batch, sampled_labels), axis=0)
            
        test_discriminator_loss = dis.evaluate(X, [y, aux_y], verbose = False)

        disc_test_loss.append(test_discriminator_loss)

        noise, sampled_labels = generate_batch_noise_and_labels(2 * nb_train, latent_dim)


        trick = np.ones(2 * nb_train)

        test_generator_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose = False
        )

        gen_test_loss.append(test_generator_loss)

        break

    discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

    generator_test_loss = np.mean(np.array(gen_test_loss), axis=0)

    discriminator_test_loss = np.mean(np.array(disc_test_loss), axis=0)

    train_history['generator'].append(generator_train_loss)
    train_history['discriminator'].append(discriminator_train_loss)

    test_history['generator'].append(generator_test_loss)
    test_history['discriminator'].append(discriminator_test_loss)

    print_logs(dis.metrics_names, train_history, test_history)

    gen.save_weights(
        'Y:/CovidGAN/parameters/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
    dis.save_weights(
        'Y:/CovidGAN/parameters/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

pickle.dump({'train': train_history, 'test':test_history},
            open('Y:/CovidGAN/acgan_history.pkl', 'wb'))
