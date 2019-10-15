import os
import glob

from unet3d.model import unet_model_3d
from data_generator import DataGenerator

DATA_LEN = 5
IMG_SIZE = 224
N_CHANNEL = 16
DATAPATH = '/home/trungdunghoang/Documents/EPFL/3DUnetCNN/data_test'

model = unet_model_3d(input_shape=(DATA_LEN, IMG_SIZE,IMG_SIZE,N_CHANNEL))

train_generator = DataGenerator(DATAPATH)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=train_generator,
                    validation_steps=len(train_generator))



