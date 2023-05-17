import math
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt




def get_white_noise(signal,SNR) :
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return noise


def Add_AWGN(signal,SNR):
    noise = get_white_noise(signal,SNR)
    noise_signal = signal+noise
    return noise_signal


##wav = r"C:\Users\bhavu\Downloads\Ch_classification\Fold_1\Data\Neutral\Test\ID018_008_Headset_Neutral_sentence_001.wav"
##
##s,fs = sf.read(wav)
##print(s)
##n = get_white_noise(s,SNR=0)
##print(n)
##s_n = Add_AWGN(s,SNR=0)
##print(s_n.shape)
##
##sf.write('new_file.wav', s_n, 16000)
####plt.plot(s)
####plt.show()
##
####plt.plot(n)
####plt.show()
##
##plt.plot(s_n)
##plt.show()

