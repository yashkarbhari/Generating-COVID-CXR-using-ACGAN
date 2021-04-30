# Function for label smoothing 

def label_smoothing(vector, max_dev = 0.2):
        d = max_dev * np.random.rand(vector.shape[0],vector.shape[1])
        if vector[0][0] == 0:
            return vector + d
        else:
            return vector - d
          
valid_o = np.ones((bs, 1))
fake_o = np.zeros((bs, 1))


