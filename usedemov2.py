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

import time

class DataSetting():
    def __init__(self):
        pass

    def cut5120(self, data):
        datalen = len(data)
        return data[0: 5120]
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

xs = np.zeros(1024)

class RealTime_VoiceIO():
    def __init__(self):
        plt.switch_backend('TkAgg')
        plt.close('all')
        self.fig= plt.figure()

        self.p_in = pa.PyAudio()
        self.fs = 16000 #サンプリング周波数
        self.channels = 1 #マイクがモノラルの場合は1にしないといけない
        self.chunk = 5120
        use_device_index = 0
        
        #formatは8bitならget_format_from_width(1)16bitは(2)
        #音源から1回読み込むときのデータサイズ。1024(=2の10乗) とする場合が多い
        self.in_stream = self.p_in.open(format = pa.paInt16,
                            channels = self.channels,
                            rate = self.fs,
                            input = True,
                            frames_per_buffer = self.chunk,
                            input_device_index = use_device_index,
                            output = True)

        # fq = np.linspace(0, self.fs, self.chunk) # 周波数軸　linspace(開始,終了,分割数)
        # self.data = np.zeros(self.chunk)
        # self.line, = plt.plot(fq ,self.data)
        # plt.ylim(0,1)
        # plt.show(block=False)
        
    # def callback(self, in_data, frame_count, time_info, status):
    #     global xs
    #     xs = np.frombuffer(in_data, dtype=np.int16).astype(np.float)
    #     return (in_data, pa.paContinue)

    def update(self):
        model = model_from_json(open('./demo/CycleGanDemoVoiceVSN/save/g_BA.json').read(), custom_objects = {'InstanceNormalization':InstanceNormalization})
        model.load_weights('./demo/CycleGanDemoVoiceVSN/save/400_133g_BA.h5')

        #history
        # model = model_from_json(open('./demo/CycleGanDemoVoiceVSN/save/randomlabel_tatamikomi_1500/g_BA.json').read(), custom_objects = {'InstanceNormalization':InstanceNormalization})
        # model.load_weights('./demo/CycleGanDemoVoiceVSN/save/randomlabel_tatamikomi_1500/g_BA.h5')

        # model = model_from_json(open('./demo/CycleGanDemoVoiceVSN/save/tatamikomi_2500/g_BA.json').read(), custom_objects = {'InstanceNormalization':InstanceNormalization})
        # model.load_weights('./demo/CycleGanDemoVoiceVSN/save/tatamikomi_2500/g_BA.h5')

        fftLen = 254
        win = np.hamming(fftLen)
        step = 78
        storage_spe = np.array([])
        comp = []
        log2500 = np.log(2500)
        log4161536 = np.log(4161536)

        while(True):
            # storage.extend(self.in_stream.read(self.chunk))
            storage = self.in_stream.read(self.chunk, exception_on_overflow=False)
            sa = np.frombuffer(storage,dtype="int16") 

            # ave = np.average(sa)
            # print(ave)
            # if (-10 <= ave and ave <= 10):
            #     continue
            # print(len(xs))

            a = DataSetting()
            # data = a.cut5120(storage)
            data = a.cut5120(sa)
            stft_data = a.stft(sa, win, step)

            abs_data = abs(stft_data[:, : fftLen // 2 + 1].T)

            Bs = []
            B = np.clip(abs_data, 2500, None)
            B = (np.log(B) - log2500) / (log4161536 - log2500) 
            B = B[:, :, np.newaxis]
            Bs.append(B)
            Bs = np.array(Bs)

            start = time.time()
            predBs = model.predict(Bs)

            backhalfspe = np.flipud(np.reshape(predBs, (128,64)))
            backhalfspe = backhalfspe[1:-1,:]
            predB = np.concatenate([predBs[0,:,:,0], backhalfspe])

            predB = np.exp(predB * (log4161536 - log2500) + log2500)
            predB[predB < 2500] = 1

            # plt.imshow(predB, cmap='gray', origin = "lower", aspect="auto")
            # plt.show()

            # storage_spe = np.append(storage_spe, predB.T)
            
            GLA = a.GriffinLim(predB.T, win, step, iterations=100)

            GLA[GLA > 32760] = 0


            # F_abs = np.abs(np.fft.fft(GLA))
            # F_abs_amp = F_abs / 2575 * 2

            # fq = np.linspace(0, 1/16000, 2575)
            # plt.plot(fq, F_abs_amp)
            # plt.show()

            # storage_spe = storage_spe[40:]

            # comp.extend(GLA)
            # if (len(comp) > 5120 * 3 * 10):
            #     break

            # del storage[0:2560]
            # return 0
            
            elapsedtime = time.time() - start
            print("time:{0}".format(elapsedtime)+ "[sec]")                   

            b = np.array(GLA, dtype='int16').tobytes()
            # output = self.in_stream.write(storage)
            output = self.in_stream.write(b)
    
        comp = np.array(comp)
        wavfile.write('./demo/CycleGanDemoVoiceVSN/demo.wav', 16000, (comp).astype(np.int16))





# PATH = './demo/CycleGanDemoVoiceVSN/materials/data/NB2/あなたが嬉しいと、わたしも嬉しいです.wav'


# rate, data = wavfile.read(PATH)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


key = input('何かを押せ')
b = RealTime_VoiceIO()
b.update()




