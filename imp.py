# -*- coding: utf-8 -*-



from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Bidirectional,LSTM,Reshape,CuDNNLSTM,BatchNormalization,Flatten,Dropout,Dense
from keras.layers import add
from keras.utils import plot_model

from keras.models import Model
import copy
import warnings
warnings.filterwarnings('ignore')
import cv2
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg') # No pictures displayed

import matplotlib.pyplot as plt
import librosa
import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment

from pydub import AudioSegment
from pydub.utils import make_chunks
#filler
from glob import glob
import cv2
import os



def resnet_model_dilation(n=2):
    ''' This model is build using keras module from the paper https://arxiv.org/pdf/1910.12590.pdf
    inputs are to be resized of 256,256*4,1 with dilation_rate
    output is the model
    '''
    input  = Input(shape = (256,256*8,1))
    bnEps=2e-5
    bnMom=0.9


    c1 = Conv2D(64, (7,7), padding='same',strides=2,activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(input)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(32, (3,3),strides=2, padding='same', use_bias=False,kernel_initializer='glorot_uniform')(input)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)

    c4 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)


    #-----------------------------------------------layer 2----------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides=2, padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(64, (3,3),strides=2, padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #----------------------------------------------layer 3------------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides = (1,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(128, (3,3),strides = (1,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 4---------------------------------------------------------------------------------

    c1 = Conv2D(64, (3,3),strides = (2,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(128, (3,3),strides = (2,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 5-----------------------------------------------------------------------------------
    c1 = Conv2D(32, (3,3),strides = (2,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(64, (3,3),strides = (2,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(32, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-----------------------------------------layer 6-------------------------------------------------------------------------
    c1 = Conv2D(16, (3,3),strides = (2,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(32, (3,3),strides = (2,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(32, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(16, (3,3), padding='same',use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    f = Flatten()(a4)
    f = Reshape((int(8192/2), 1))(f)

    # #-----------------------------------------layer7---------------------------------------------------------------------------
    bi1 = Bidirectional(CuDNNLSTM(512, return_sequences=True))(f)
    d1  = Dropout(0.2)(bi1)

    bi2 = Bidirectional(CuDNNLSTM(512))(d1)
    d2 = Dropout(0.4)(bi2)

    out = Dense(n,activation='softmax')(d2)

    # create model
    model = Model(inputs=input, outputs=out)
    return model





def superimpose(img, cam):
    """superimpose original image and cam heatmap"""

    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * .5 + img * .5
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return img, heatmap, superimposed_img

def plot(model, cam_func, img):

  i = img
  img = img/255.0
  x = np.expand_dims(img, axis=0)
  # x = preprocess_input(copy.deepcopy(x))


    # cam / superimpose
  cls_pred, cam = cam_func(model=model, x=x, layer_name=model.layers[-9].name)
  img, heatmap, superimposed_img = superimpose(i, cam)

  fig, axs = plt.subplots(ncols=3, figsize=(18,3))

  axs[0].imshow(i.squeeze())
  axs[0].set_title('original image')
  axs[0].axis('off')

  axs[1].imshow(heatmap)
  axs[1].set_title('heatmap')
  axs[1].axis('off')

  axs[2].imshow(superimposed_img)
  axs[2].set_title('superimposed image')
  axs[2].axis('off')

  plt.suptitle(' / Predicted label : ' + class_to_label[cls_pred])
  plt.tight_layout()
  plt.show()

def grad_cam(model, x, layer_name):
    """Grad-CAM function"""

    cls = np.argmax(model.predict(x))

    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    # Get outputs and grads
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([x])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1)) # Passing through GlobalAveragePooling

    cam = np.dot(output, weights) # multiply
    cam = np.maximum(cam, 0)      # Passing through ReLU
    cam /= np.max(cam)            # scale 0 to 1.0

    return cls, cam




def mel_spectrogram(self,save_path,max_frequency):
    '''
    inputs self,save_path,frequency limits,save
    saves a image as output
    '''
    plt.figure(figsize=(14, 5))
#     X = librosa.stft(self.signalData)
#     Xdb = librosa.amplitude_to_db(abs(X))
    y,sr = self.signalData,self.samplingFrequency

    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

    librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=max_frequency)

    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()

class audio:
    def __init__(self,path):
        self.signalData,self.samplingFrequency = librosa.load(path,sr = 22500)






def clean_dir(path):
    files_di = glob(f'{path}/*')
    for f in files_di:
        os.remove(f)




def make_4sec(audio_path,n = 4):
    myaudio = AudioSegment.from_file(audio_path , "wav")
    chunk_length_ms = n*1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of four sec

    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = "/content/melspectrograms/chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

        f = audio(chunk_name)
        mel_spectrogram(f,f'/content/melspectrograms/{i}.png',max_frequency=2000)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
label_to_class = {
    'filler': 0,
    'nonfiller':1
}
class_to_label = {v: k for k, v in label_to_class.items()}



def read_img(file_path):
    img_spec = cv2.imread(file_path,0)
    img_spec = cv2.resize(img_spec,(256*4*2,256))
    img_spec=np.expand_dims(img_spec,axis = -1)
    return img_spec

def plot_cam_grad(model):
    images = glob('/content/melspectrograms/*.png')
    for i in sorted(images):
        print(i)
        img = read_img(i)
        plot(model=model, cam_func=grad_cam, img=img)
        print(' '.center(150,'-'))
        
def plot_cam_grad_result(model,image):    
    print(image)
    img = read_img(image)
    plot(model=model, cam_func=grad_cam, img=img)
