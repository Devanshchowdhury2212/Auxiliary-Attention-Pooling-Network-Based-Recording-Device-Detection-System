import tensorflow as tf
from tensorflow import keras
#import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
#from TDNNLayer import TDNNLayer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, AveragePooling1D, Concatenate,Activation, GlobalAveragePooling1D, AveragePooling2D, Flatten, GlobalMaxPooling1D, Lambda, LSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
#import h5py
import time
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.callbacks
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Identity, Constant,RandomNormal, Zeros
import os
import math
import scipy.io

def GetModel(filters=20, kernel_size=(3,4)):
       	F=24;
        #ivec = Input(batch_shape=(1,100));
       	inp=Input(batch_shape=(1,None,F,1));
        #wts1 = AveragePooling2D(pool_size=(4, 1), strides= None, padding='valid')(inp)
        inp1 = Reshape((-1,24))(inp)
        #inp2 = AveragePooling1D(pool_size= 4, strides= None, padding='valid')(inp1)
        #wts = TimeDistributed(Dense(1,activation='sigmoid'))(inp1)
        wts1 = LSTM(24,return_sequences=True)(inp1)
        wts = LSTM(1,return_sequences=True)(wts1)
        wts_out = Activation('sigmoid')(wts)
        inp2 = AveragePooling1D(pool_size= 4, strides= None, padding='valid')(wts_out)
        cnn_1=Conv2D(filters, kernel_size, padding='same', activation='relu')(inp)
        cnn_1_n=BatchNormalization()(cnn_1)
        cnn11_n = AveragePooling2D(pool_size=(2, 1), strides= None, padding='valid')(cnn_1_n)
        cnn_2=Conv2D(filters, kernel_size, strides=1, padding='same', activation='relu')(cnn11_n)
        cnn_2_n=BatchNormalization()(cnn_2)
        cnn_22_n = AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid')(cnn_2_n)                                                                                  
        cnn_3=Conv2D(20, kernel_size, strides=1, padding='same', activation='relu')(cnn_22_n)
        cnn_3_n=BatchNormalization()(cnn_3)
        cnn_4 = AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid')(cnn_3_n)
        #cnn_5=Conv2D(30, kernel_size, strides=1, padding='same', activation='relu')(cnn_4)
        #cnn_5_n=BatchNormalization()(cnn_5)
        #cnn_6=Conv2D(30, kernel_size, strides=1, padding='same', activation='relu')(cnn_5_n)
        #cnn_6_n=BatchNormalization()(cnn_6)
        cnn   = Reshape((-1,240))(cnn_4)
        dnn = Lambda(mult)([cnn,inp2])
        #lstm_4= CuDNNLSTM(60, return_sequences=True)(dnn)
                                
        #lstm_5= CuDNNLSTM(80,return_sequences=True)(cnn)                   
        #illum_est = Lambda(norm)(lstm_5);
        layer_5_n = GlobalAveragePooling1D()(dnn)
        #layer_2= Dense(100,activation='linear')(ivec)
        
        layer_5= Dense(120,activation='relu')(layer_5_n)
        #layer_con = Concatenate()([layer_5,layer_2])
        
        #layer_5_n=BatchNormalization()(layer_5)
        #layer_5_1 = Dense(300,activation='relu')(layer_5_n)
        #tf.tensorflow.keras.layers.
        layer_6= Dense(5,activation='softmax')(layer_5)

        return(inp,layer_6)
           
        
         
           
def norm(fc2):
    cnn = K.squeeze(fc2, axis = 3)
    #tf.convert_to_tensor(arg, dtype=tf.float32)
    return cnn
              
def mult(d1):
    #k = tf.ones(tf.shape(d1[1]))
    mu = tf.math.multiply(d1[0],d1[1])   #tf.math.subtract(k,d1[1]))
    #tf.convert_to_tensor(arg, dtype=tf.float32)
    return mu




        
    
    
