import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, LSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
import time
import soundfile as sf
from features import mfcc, fbank, logfbank
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.callbacks
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Identity, Constant,RandomNormal, Zeros
import os
import math
import scipy.io
from tensorflow.keras.layers import Layer
from model_fear1 import GetModel

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

Data_train = './Fold_1/Data/Neutral/'
def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data

F=24;
files1 = np.genfromtxt('./Fold_1/lists/train_data_n.lst', dtype='str')
list_IDs_in=[str(x) for x in files1]

inp,out=GetModel()
model=Model(inp,out);
model.load_weights('./Fold_1/Models_n/checkpoint-phn_emb-06-0.9631.hdf5');
layer_name = 'lstm_1'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

for i in range(0,len(list_IDs_in)):
    ID = list_IDs_in[i]
    [s,fs] = sf.read(Data_train + ID)
    x1 = logfbank(s,fs)
    x2 = np.zeros(shape = x1.shape)
    x2 = x1 - x1.mean(0)
    x = x2.reshape((1,x2.shape[0],x2.shape[1],1))
    Attention_weights = intermediate_layer_model.predict(x)
    final_output = model.predict(x)
    mat = ID[5:]+'.mat'
    file_path = r'C:/Users/bhavu/Downloads/Ch_classification/Fold_1/Train_Attention/Attention_neutral'+'/'+mat
    scipy.io.savemat(file_path,{'final_output':final_output,'signal':s,'Attention_weights':Attention_weights})













