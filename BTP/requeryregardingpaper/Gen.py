import os
import numpy as np

##path = r'D:\All_88\Whisper\Val_2'
##
##ids = []
##for i in os.listdir(path):
##    ID = i.split("_")[0]
##    ids.append(ID)
##
##dr = list(set(ids))
##print(len(dr))

##path = r'C:\Users\bhavu\Downloads\Ch_classification\Fold_2\Data\Whisper\Train'
##
##x = []
##y = []
##for i in os.listdir(path):
##    x.append('Train/'+i)
##    l = i.split("_")[2]
##    if (l == 'Zoom'):
##        y.append(int(0))
##    if (l == 'Iphone'):
##        y.append(int(1))
##    if (l == 'Moto'):
##        y.append(int(2))
##    if (l == 'Headset'):
##        y.append(int(3))
##    if (l == 'Nokia'):
##        y.append(int(4))
##
##x1 = np.array(x, dtype='str')
##y1 = np.array(y, dtype='int')
##np.savetxt('train_data_w.lst', x1,fmt="%s")
##np.savetxt('train_label_w.lst', y1)
    
##files1 = np.genfromtxt('./Fold_2/lists/train_data_n.lst', dtype='str')
##list_IDs_in=[str(x) for x in files1]
##labels = np.genfromtxt('./Fold_2/lists/train_label_n.lst', dtype='int32')
##print(labels)
##print(list_IDs_in)
