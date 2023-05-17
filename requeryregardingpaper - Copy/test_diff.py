import tensorflow as tf
from tensorflow import keras
import numpy as np
import soundfile as sf
from features import mfcc, fbank, logfbank
from tensorflow.keras.models import Model
import os
import scipy.io
from model_fear1 import GetModel
from sklearn.metrics import accuracy_score, confusion_matrix
from noise import Add_AWGN


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

files1 = np.genfromtxt('./Fold_1/lists/test_data_w.lst', dtype='str')
labels = np.genfromtxt('./Fold_1/lists/test_label_w.lst', dtype='int32')


list_IDs_in=[str(x) for x in files1]

inp,out=GetModel()
model=Model(inp,out);
model.load_weights('./Fold_1/Models_n/checkpoint-phn_emb-06-0.9631.hdf5')


y_true = []
y_pred = []

for i in list_IDs_in:
    wav = './Fold_1/Data/Whisper/' + i
    ind = list_IDs_in.index(i)
    #print(i,ind,labels[ind])
    [s,fs] = sf.read(wav)

    #s_n = Add_AWGN(s,SNR=5)
    
    x1 = logfbank(s,16000)
    x2 = np.zeros(shape = x1.shape)
    x2 = x1 - x1.mean(0)
    x = x2.reshape((1,x2.shape[0],x2.shape[1],1))
    pred = model.predict(x)
    true = labels[ind]
    y_pred.append(np.argmax(pred,axis=1)[0])
    y_true.append(true)
    print(np.argmax(pred,axis=1))
    print(true)




acc = accuracy_score(y_true, y_pred)
print("Test_accuracy ----> " + str(acc))
print(confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4]))

