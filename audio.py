from scipy.io import wavfile
import numpy as np
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import scipy
from pydub import AudioSegment
import os

class audio:
    def __init__(self,file):
        self.path  = file
        self.name = file.split('/')[-1].split('.')[0]
        self.signalData,self.samplingFrequency  = librosa.load(self.path)
        self.duration = librosa.get_duration(filename=self.path)

def plot_spectrogram(self,name,limits = (0,10000),save=False,overlap_f0 = False,cat = 'False'):
    plt.figure(figsize=(14, 5))
    X = librosa.stft(self.signalData)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=self.samplingFrequency,cmap = 'magma')
    l1,l2 = limits
    plt.ylim(l1,l2)
    plt.axis('off')
    if save == True:
        if cat == False:
            print('spectrograms/nonfillers-'+str(name)+'.png')
            plt.savefig('spectrograms/nonfillers/nf-'+str(name)+'.png')
        
        if cat == True:
            print('spectrograms/fillers-'+str(name)+'.png')
            plt.savefig('spectrograms/fillers/f-'+str(name)+'.png')