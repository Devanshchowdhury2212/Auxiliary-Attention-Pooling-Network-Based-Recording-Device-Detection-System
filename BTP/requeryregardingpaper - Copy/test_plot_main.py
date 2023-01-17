import tensorflow as tf
from tensorflow import keras
#import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import     
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
#from TDNNLayer import TDNNLayer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, LSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
#import h5py
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
#import tensorflow.contrib.eager as tfe
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#root='./ITS/';
Data_train = './Fold_2/Data/Whisper/'
#Data_ivec = './ivectors/'
def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data
F=24;
files1 = np.genfromtxt('./Fold_2/lists/test_data_w.lst', dtype='str')
list_IDs_in=[str(x) for x in files1]
#labels = np.genfromtxt('./split_1_label.lst', dtype='int32')
#mask=Input(batch_shape=(32,1500,1));
#inp=Input(batch_shape=(1,1500,F,1));
#layer1 = TDNNLayer(context=[-2,2],input_dim=24,output_dim=512,activation = 'relu', full_context = True)(inp)
#layer6 = MaskPooling()([inp,mask]);
inp,out=GetModel()
model=Model(inp,out);
model.load_weights('./Fold_2/Models_w/checkpoint-phn_emb-09-0.9700.hdf5');
layer_name = 'lstm_1'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
#pred_split1 = np.zeros((4206,1,183))
#for i in range(0,4206):

ind = list_IDs_in.index('Test/ID0090_001_Headset_Whisper_sentence_0021.wav')
#print(ind)
ID = list_IDs_in[ind]
print(ID)
[s,fs] = sf.read(Data_train + ID)
x1 = logfbank(s,fs)
x2 = np.zeros(shape = x1.shape)
x2 = x1 - x1.mean(0)

x = x2.reshape((1,x2.shape[0],x2.shape[1],1))
  
Attention_weights = intermediate_layer_model.predict(x)
final_output = model.predict(x);
#print(s.shape)
#print(Attention_weights.shape)
print(np.argmax(final_output,axis=1))

#file_path = r'C:\Users\bhavu\Downloads\Ch_classification\Attention_w\ID018_008_sen_10/Attention_pred_Headset.mat'
#scipy.io.savemat(file_path,{'final_output':final_output,'signal':s,'Attention_weights':Attention_weights})



