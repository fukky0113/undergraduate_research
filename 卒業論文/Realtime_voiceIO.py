from keras.models import load_model, model_from_json
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend 

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import scipy
from scipy.io import wavfile

import pyaudio as pa
import sys

class DataSetting():
    def __init__(self):
        pass

    def stft(self, x, win, step):
        l = len(x)
        N = len(win)
        M = int(np.ceil(float(l - N + step)/ step))

        new_x = np.zeros(N + ((M - 1) * step), dtype = np.float64)
        new_x[: l] = x 
        X = np.zeros([M, N], dtype = np.complex64)
        for m in range(M):
            start = step * m
            X[m, :] = np.fft.fft(new_x[start : start + N] * win)

        return X
    
    def istft(self, X, win, step):
        M, N = X.shape
        assert (len(win) == N),
        l = (M - 1) * step + N
        x = np.zeros(l, dtype = np.float64)
        wsum = np.zeros(l, dtype = np.float64)
        for m in range(M):
            start = step * m
            x[start : start + N] = x[start : start + N] + np.fft.ifft(X[m, :]).real * win
            wsum[start : start + N] += win ** 2
        pos = (wsum != 0)
        x_pre = x.copy()
        x[pos] /= wsum[pos]
        return x

    def GriffinLim(self, a, win, step, iterations=50):
        approximated_signal=None
        for k in range(iterations):
            if approximated_signal is None:
                _P = np.random.randn(*a.shape)
            else:
                _D = self.stft(approximated_signal, win, step)
                _P = np.angle(_D)
            _D = a * np.exp(1j * _P)
            approximated_signal = self.istft(_D, win, step)
        return approximated_signal

class RealTime_VoiceIO():
    def __init__(self):
        self.p_in = pa.PyAudio()
        self.fs = 16000
        self.channels = 1
        self.chunk = 5120
        use_device_index = 1

        self.in_stream = self.p_in.open(format = pa.paInt16,
                            channels = self.channels,
                            rate = self.fs,
                            input = True,
                            frames_per_buffer = self.chunk,
                            input_device_index = use_device_index,
                            output = True)

    def update(self):
        model = model_from_json(open('モデル:jsonファイル').read())
        model.load_weights('重み:h5ファイル')

        fftLen = 254
        win = np.hamming(fftLen)
        step = 62
        Min = 1000
        log1000 = np.log(Min)
        log4161536 = np.log(4161536)
        dataset = DataSetting()

        while(True):
            read_data = self.in_stream.read(self.chunk, exception_on_overflow=False)
            read_data = np.frombuffer(read_data,dtype="int16") 
            stft_data = dataset.stft(read_data, win, step)
            abs_data = abs(stft_data[:, : fftLen // 2 + 1].T)
            Bs = []
            B = np.clip(abs_data, A, None)
            B = (np.log(B) - log1000) / (log4161536 - log1000) 
            B = B[:, :, np.newaxis]
            Bs.append(B)
            Bs = np.array(Bs)
            predBs = model.predict(Bs.T)
            predBs = predBs.T
            backhalfspe = np.flipud(np.reshape(predBs, (128,80)))
            backhalfspe = backhalfspe[1:-1,:]
            pred = np.concatenate([predBs[0,:,:,0], backhalfspe])
            pred = np.exp(pred * (log4161536 - log1000) + log1000)
            pred[pred < A] = 1
            pred = np.pad(pred, [(0,0), (10,10)], 'constant')
            GLA = dataset.GriffinLim(pred.T, win, step, iterations=70)
            GLA = GLA[804:-804]
            b = np.array(GLA, dtype='int16').tobytes()
            output = self.in_stream.write(b)

if __name__ == "__main__":
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    RealTime_VoiceIO = RealTime_VoiceIO()
    RealTime_VoiceIO.update()

