from keras.models import load_model, model_from_json
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend 

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile

import pyaudio as pa
import sys

class DataSetting():
    def __init__(self):
        pass

    def cut5120(self, data):
        datalen = len(data)
        return data[0: 2560]
        # random = np.random.randint(0, datalen - 2560)
        # return data[random : random + 2560]


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
        assert (len(win) == N), "FFT length and window length are different."
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


    def GriffinLim(self, a, win, step, iterations=100):
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

xs = np.zeros(1024)

class RealTime_VoiceIO():
    def __init__(self):
        plt.switch_backend('TkAgg')
        plt.close('all')
        self.fig= plt.figure()

        self.p_in = pa.PyAudio()
        self.fs = 16000 #サンプリング周波数
        self.channels = 1 #マイクがモノラルの場合は1にしないといけない
        self.chunk = 2560
        use_device_index = 0
        
        #formatは8bitならget_format_from_width(1)16bitは(2)
        #音源から1回読み込むときのデータサイズ。1024(=2の10乗) とする場合が多い
        self.in_stream = self.p_in.open(format = pa.paInt16,
                            channels = self.channels,
                            rate = self.fs,
                            input = True,
                            frames_per_buffer = self.chunk,
                            input_device_index = use_device_index,
                            stream_callback=self.callback)

        # fq = np.linspace(0, self.fs, self.chunk) # 周波数軸　linspace(開始,終了,分割数)
        # self.data = np.zeros(self.chunk)
        # self.line, = plt.plot(fq ,self.data)
        # plt.ylim(0,1)
        # plt.show(block=False)
        
    def callback(self, in_data, frame_count, time_info, status):
        global xs
        xs = np.frombuffer(in_data, dtype=np.int16).astype(np.float)
        return (in_data, pa.paContinue)

    def update(self):
        model = model_from_json(open('./demo/CycleGanDemoVoiceV2/save/g_BA.json').read(), custom_objects = {'InstanceNormalization':InstanceNormalization})
        model.load_weights('./demo/CycleGanDemoVoiceV2/save/g_BA.h5')
        fftLen = 128 - 2
        win = np.hamming(fftLen)
        step = fftLen // 4
        storage = []
        comp = []

        while(True):
            
            storage.extend(xs)
            # F = np.fft.fft(xs)
            # F_abs = np.abs(F)
            # F_abs_amp = F_abs / self.chunk * 2

            print(len(xs))

            a = DataSetting()
            data = a.cut5120(storage)
            stft_data = a.stft(data, win, step)

            abs_data = abs(stft_data[:, : fftLen // 2 + 1].T)

            Bs = []
            B = np.clip(abs_data, 2500, None)
            B = (np.log(B) - np.log(2500)) / (np.log(4161536) - np.log(2500)) 
            B = B[:, :, np.newaxis]
            Bs.append(B)
            Bs = np.array(Bs)

            predBs = model.predict(Bs)

            backhalfspe = np.flipud(predBs[0,:,:,0])
            backhalfspe = backhalfspe[1:-1,:]
            predB = np.concatenate([predBs[0,:,:,0], backhalfspe])

            predB = np.exp(predB * (np.log(4161536) - np.log(2500)) + np.log(2500))
            predB[predB < 2500] = 1

            # plt.imshow(predB, cmap='gray', origin = "lower", aspect="auto")
            # plt.show()

            GLA = a.GriffinLim(predB.T, win, step)

            # self.line.set_ydata(GLA)
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            comp.extend(GLA)

            if (len(comp) > 20000):
                break
            del storage[0:2560]
            # return 0

        comp = np.array(comp)
        wavfile.write('./demo/CycleGanDemoVoiceV2/demo.wav', 16000, (comp).astype(np.int16))




# PATH = './demo/CycleGanDemoVoiceV2/materials/data/NB2/あなたが嬉しいと、わたしも嬉しいです.wav'


# rate, data = wavfile.read(PATH)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


key = input('続けるには y を入力してください。')
b = RealTime_VoiceIO()
b.update()




