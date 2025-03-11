import numpy as np
import glob
import scipy
from scipy.io import wavfile


appendData = []
paths = glob.glob('C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NB2-myvoice2/*.wav')
for path in paths:
    print(path)
    rate, data = wavfile.read(path)
    appendData.extend(data)
appendData = np.array(appendData)
wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NB2-myvoice3/allAppend.wav', 16000, (appendData).astype(np.int16))
