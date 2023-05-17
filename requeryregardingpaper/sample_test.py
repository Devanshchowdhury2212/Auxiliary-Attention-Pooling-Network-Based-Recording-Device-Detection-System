import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
#from TDNNLayer import TDNNLayer
from keras.layers import Input
from keras.layers import Conv1D, Conv2D,MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, CuDNNLSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
#import h5py
import time
import soundfile as sf
from features import mfcc, fbank, logfbank
from keras.models import Model
from keras.utils import Sequence
import keras.callbacks
import keras.backend as K
from keras.initializers import Identity, Constant,RandomNormal, Zeros
import os
import math
import scipy.io
from keras.layers import Layer
from model_fear1 import GetModel
#import tensorflow.contrib.eager as tfe
os.environ["CUDA_VISIBLE_DEVICES"]="1"
root='./ITS/';
Data_train = './'
Data_ivec = './ivectors/'
def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data
F=24;
files1 = np.genfromtxt('./vad_dev.lst', dtype='str')
list_IDs_in=[str(x) for x in files1]
#labels = np.genfromtxt('./split_1_label.lst', dtype='int32')
#mask=Input(batch_shape=(32,1500,1));
#inp=Input(batch_shape=(1,1500,F,1));
#layer1 = TDNNLayer(context=[-2,2],input_dim=24,output_dim=512,activation = 'relu', full_context = True)(inp)
#layer6 = MaskPooling()([inp,mask]);
inp,out=GetModel()
model=Model(inp,out);
model.load_weights('./Models/checkpoint-phn_emb-15-3.14.hdf5');
layer_name = 'cu_dnnlstm_2'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
#pred_split1 = np.zeros((4206,1,183))
for i in range(0,1):

   #ID = list_IDs_in[i]
   [s,fs] = sf.read('./sample_data/frf03_f01_whsp.wav')
   intr_out = np.zeros((math.floor(np.size(s)/80000),500))
   for j in range(0,math.floor(np.size(s)/80000)-1):
      s1 = s[(j*80000):((j+1)*80000)+82]  #(np.minimum(((j+1)*80000)+81,np.size(s)))]
      #print(np.size(s1))
      x1 = logfbank(s1,fs)
      
      x2 = np.zeros(shape = x1.shape)
      x2 = x1 - x1.mean(0)
   #y = labels[i]
      x = x2.reshape((1,x2.shape[0],x2.shape[1],1))
  
      intermediate_output = intermediate_layer_model.predict(x)
      intr_out[j] = np.squeeze(intermediate_output)
      print(j)
#pred_split1 = model.predict(x);
  
  
   print(i)
   print('Comp')
#scipy.io.savemat('spec_pred1.mat',{'pred_split1':pred_split1,'s':s,'intr':intermediate_output})
   scipy.io.savemat('./VAD/vad_pred_'+str(i+1).zfill(2)+'.mat',{'s':s,'fs':fs,'intr':intr_out})
#scipy.io.savemat('/home/abinay/Documents/DNN/norm11.mat', {'mfcc_norm':nr_ls})


