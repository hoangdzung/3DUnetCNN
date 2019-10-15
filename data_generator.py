import numpy as np
import keras
import os 
from scipy.io import loadmat
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, datapath, batch_size=1, data_length=5, img_size=224, n_channels=16, shuffle=True):
    # def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
    #              n_classes=10, shuffle=True):
        'Initialization'
        self.datadir = datapath
        self.batch_size = batch_size
        self.data_length = data_length
        self.img_size = img_size
        self.n_channels = n_channels
        self.mat_files = sorted(os.listdir(datapath))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.mat_files) -self.data_length) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.mat_files[self.data_length:]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X_batch = []
        y_batch = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X = []
            for j, mat_file in enumerate(self.mat_files[ID:ID+self.data_length]):
                X_step, _ = self.__read_data_from_mat(os.path.join(self.datadir, mat_file))
                X.append(X_step)
            _, y = self.__read_data_from_mat(os.path.join(self.datadir, self.mat_files[ID+self.data_length]))
            y = np.expand_dims(np.expand_dims(y,-1),0)
            X = np.stack(X)

            _,hx,wx,_ = X.shape
            _,hy,wy,_ = y.shape 

            assert hx==hy and wx==wy

            startx = random.randint(0, hx-self.img_size-1)
            starty = random.randint(0, hy-self.img_size-1)

            X = X[:,startx:startx+self.img_size, starty: starty+self.img_size]
            y = y[:,startx:startx+self.img_size, starty: starty+self.img_size]

            X_batch.append(X)
            y_batch.append(y)
        
        X_batch = np.stack(X_batch)
        y_batch = np.stack(y_batch)

        return X_batch, y_batch

    def __read_data_from_mat(self, path):
        data = loadmat(path)
        return data['image'], data['gt']

if __name__ == '__main__':
    datapath = '/home/trungdunghoang/Documents/EPFL/3DUnetCNN/data_test'
    datagen = DataGenerator(datapath, batch_size=5)
