##import tensorflow as tf
##from tensorflow import keras
###import tensorflow_probability as tfp
import numpy as np
##from tensorflow.keras.layers import Input
##from tensorflow.keras.layers import Dense, Dropout, Activation
##from tensorflow.keras.layers import LSTM
###from TDNNLayer import TDNNLayer
##from tensorflow.keras.layers import Input
##from tensorflow.keras.layers import Conv1D, Conv2D,MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, LSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
###import h5py
##import time
##import soundfile as sf
##from features import mfcc, fbank, logfbank
##from tensorflow.keras.models import Model
##from tensorflow.keras.utils import Sequence
##import tensorflow.keras.callbacks
##import tensorflow.keras.backend as K
##from tensorflow.keras.initializers import Identity, Constant,RandomNormal, Zeros
import os
##import math
import scipy.io
#from tensorflow.keras.layers import Layer
#from model_fear1 import GetModel
from sklearn.metrics import accuracy_score, confusion_matrix


files1 = np.genfromtxt('./Fold_2/lists/test_data_n.lst', dtype='str')
labels = np.genfromtxt('./Fold_2/lists/test_label_n.lst', dtype='int32')
list_IDs_in=[str(x) for x in files1]

path = r'C:\Users\bhavu\Downloads\Ch_classification\Fold_2\Attention_neutral'

y_true = []
y_pred = []

for i in os.listdir(path):
    mat = scipy.io.loadmat(os.path.join(path,i))
    pred = mat['final_output']
    wav = str('Test/'+i[:-4])
    ind = list_IDs_in.index(wav)
    true = labels[ind]
    y_pred.append(np.argmax(pred,axis=1)[0])
    y_true.append(true)
    print(np.argmax(pred,axis=1))
    print(true)

acc = accuracy_score(y_true, y_pred)
print("Test_accuracy ----> " + str(acc))
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]))

















