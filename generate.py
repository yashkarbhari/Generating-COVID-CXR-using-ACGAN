import cv2

filename = 'Y:/CovidGAN/parameters/params_generator_epoch_1199.hdf5'
gen = generator(latent_dim = 100, n_classes = 2)
gen.load_weights(filename)

noise, sampled_labels = generate_batch_noise_and_labels(batch_size=62, latent_dim=100)
print(sampled_labels[0])

generated_images = gen.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)

norm_image = cv2.normalize((generated_images + 1) * 127.5, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
norm_img = norm_image.astype(np.uint8)
img = norm_img[0,:,:]
print(img.shape)

plt.imshow(img, cmap='gray')
_ = plt.axis('off')
