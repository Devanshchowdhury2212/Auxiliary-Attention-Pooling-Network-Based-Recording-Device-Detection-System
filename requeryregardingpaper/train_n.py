import tensorflow as tf
from tensorflow import keras
import sys
#import tensorflow_probability as tfp
import numpy as np
#from PIL import Image
#import cv2 as cv
import scipy.misc
from features import mfcc, fbank, logfbank
import soundfile as sf
#from i_vec_layer import i_vec_layer
from sklearn.utils import class_weight
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
import h5py
import time
import tensorflow.keras
import scipy.io
from scipy.signal import decimate
#from model_contrast import GetModel, modl
from scipy.io import wavfile, loadmat
from scipy.signal import lfilter
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.callbacks
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Identity, Constant,RandomNormal, Zeros
import os
import scipy.io
from model_fear1 import GetModel
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##os.environ["CUDA_VISIBLE_DEVICES"]="0"
##expts_dir='./Models/'
#root='./ITS/';
Data_train = './Fold_2/Data/Neutral/'
#Data_ivec = './ivectors/'
def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data
class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for tensorflow.keras'
    def __init__(self, file_name_in, file_name_out, batch_size=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.files1 = np.genfromtxt(file_name_in, dtype='str')
        self.list_IDs_in=[str(x) for x in self.files1]
        self.labels = np.genfromtxt(file_name_out, dtype='int32')
        #self.list_IDs_out=[str(x) for x in self.files2]
        #self.list_IDs_out = list_IDs_out
        #self.list_IDs_in = list_IDs_in
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        #print(len(self.list_IDs_in.shape))
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs_in)
        
    def __getitem__(self, idx):
        'Generate one batch of data'
        ID = self.list_IDs_in[idx]
        [s,fs] = sf.read(Data_train + ID)
        if fs != 16000:
            l1 = int(len(s)/fs)
            s = scipy.signal.resample(s, l1*16000, t=None, axis=0, window=None)
        x1 = logfbank(s,16000)
        x2 = np.zeros(shape = x1.shape)
        x2 = x1 - x1.mean(0)
        y = self.labels[idx]
        x = x2.reshape((1,x2.shape[0],x2.shape[1],1))
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        #list_IDs_temp_in = [self.list_IDs_in[k] for k in indexes]
        #list_IDs_temp_out = [self.list_IDs_out[k] for k in indexes]
        #print(list_IDs_temp_in)
        # Generate data
        #X, y = self.__data_generation(list_IDs_temp_in)
        y1 = tensorflow.keras.utils.to_categorical(y, 5).reshape((1,5))
        return x, y1               #tensorflow.keras.utils.to_categorical(y, 183)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs_in))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp_in):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #x = np.empty((self.batch_size))
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp_in):
            # Store sample
            [s,fs] = sf.read(Data_train + ID)
            
            x1 = logfbank(s)
            x2 = np.zeros(shape = x1.shape)
            x2 = x1 - x1.mean(0)
            #print(x2.shape)
            #x[i,] = x2
            #print(ID)   
            y = self.labels[ID]
        #print(i)    
        #return x,tensorflow.keras.utils.to_categorical(y, 6112)
        x = x2.reshape((1,x2.shape[0],x2.shape[1],1))    
            
        return x,tensorflow.keras.utils.to_categorical(y, 5)        #tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

start=time.time();
inp,out=GetModel()
model=Model(inp,out);
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']);
#model.load_weights('./Models/checkpoint-phn_emb-06-1.44.hdf5');
model.summary();
labels1 = np.genfromtxt('./Fold_2/lists/train_label_n.lst', dtype='int32')
#class_weight = class_weight.compute_class_weight('balanced', np.unique(labels1), labels1)
#print(class_weight)
callback1=tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0,patience=4,verbose=0, mode='auto')
callback2=tensorflow.keras.callbacks.ModelCheckpoint('./Fold_2/Models_n/checkpoint-phn_emb-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
print('..fit model')	
history=model.fit_generator(DataGenerator('./Fold_2/lists/train_data_n.lst','./Fold_2/lists/train_label_n.lst'),epochs=10,validation_data=DataGenerator('./Fold_2/lists/val_data_n.lst','./Fold_2/lists/val_label_n.lst'),callbacks=[callback1,callback2], workers=20, use_multiprocessing=False)
