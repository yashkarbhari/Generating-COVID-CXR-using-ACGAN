# Function for label smoothing 

def label_smoothing(vector, max_dev = 0.2):
        d = max_dev * np.random.rand(vector.shape[0],vector.shape[1])
        if vector[0][0] == 0:
            return vector + d
        else:
            return vector - d
          
valid_o = np.ones((bs, 1))
fake_o = np.zeros((bs, 1))

# Function for printing the logs after every epoch

def print_logs(metrics_names, train_history, test_history):

    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
        'component', *metrics_names))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
    print(ROW_FMT.format('generator (train)',
                         *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                         *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                         *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                         *test_history['discriminator'][-1]))

# Function to generate a batch of noise and label 

def generate_batch_noise_and_labels(batch_size, latent_dim):

    # generate a new batch of noise
    noise = np.random.uniform(-1, 1, (batch_size, latent_dim))

    # sample some labels
    sampled_labels = np.random.randint(0, 2, batch_size)

    return noise, sampled_labels

# Function to generate the loss graph using the saved weights

import pandas as pd
path = 'Y:/CovidGAN/acgan_history.pkl'
path.encode('utf-8').strip()
hist = pickle.load(open(path, 'rb'))

for p in ['train', 'test']:
    for g in ['discriminator', 'generator']:
        hist[p][g] = pd.DataFrame(hist[p][g], columns=['loss', 'generation_loss', 'auxiliary_loss'])
        plt.plot(hist[p][g]['generation_loss'], label='{} ({})'.format(g, p))

# get the NE and show as an equilibrium point
plt.hlines(-np.log(0.5), 0, hist[p][g]['generation_loss'].shape[0], label='Nash Equilibrium')
plt.legend()
plt.title(r'$L_s$ (generation loss) per Epoch')
plt.xlabel('Epoch')
plt.ylabel(r'$L_s$')
plt.show()
