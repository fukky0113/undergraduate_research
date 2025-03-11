import numpy as np
import glob
import scipy
from scipy.io import wavfile


devideData = []
path = 'C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NB2-myvoice3/allAppend.wav'
rate, data = wavfile.read(path)

for i in range(len(data)//16000):
    devideData = data[i*16000 :(i+1)*16000]
    devideData = np.array(devideData)
    wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NB2-myvoice3/%dto%dsec.wav' % (i, i + 1), 16000, (devideData).astype(np.int16))
