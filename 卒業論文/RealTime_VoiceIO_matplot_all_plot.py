from keras.models import load_model, model_from_json
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend 

from keras.preprocessing.image import img_to_array, load_img
import matplotlib

import pyaudio as pa
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import ceil, complex64, float64, hanning, zeros

import time

# xs = np.zeros(1024)

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


        # 上の段のグラフ
        # self.ax_up = plt.subplot(3, 1, 1)
        # self.realtime_line, = self.ax_up.plot(np.zeros(self.chunk))
        # plt.ylim(-1,1)

        # 中の段のグラフ
        self.ax_mid = plt.subplot(2, 1, 1)   
        # self.spe = self.ax_mid.imshow(np.zeros((257, 805)), aspect = "auto", origin = "lower", animated=True)#, cmap='gray_r')
        self.spe = self.ax_mid.imshow(np.zeros((128, 80)),aspect = "auto", origin = "lower", animated=True,vmin=0, vmax=1)#, cmap='gray_r')


        # 下の段のグラフ
        self.ax_bot = plt.subplot(2, 1, 2)   
        # self.spe = self.ax_mid.imshow(np.zeros((257, 805)), aspect = "auto", origin = "lower", animated=True)#, cmap='gray_r')
        self.spe2 = self.ax_bot.imshow(np.zeros((128, 80)),aspect = "auto", origin = "lower", animated=True,vmin=0, vmax=1)#, cmap='gray_r')

        plt.show(block=False)

    #短時間フーリエ変換(データ,長さ?フレーム?用語の名前が分からん)
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
    
    # def callback(self, in_data, frame_count, time_info, status):
    #     global xs
    #     xs = np.frombuffer(in_data, dtype=np.int16).astype(np.float) / 32768
    #     return (in_data, pa.paContinue)

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

    def update(self):
        model = model_from_json(open('A:\アンケート\アンケートいいやつ\g_BA.json').read(), custom_objects = {'InstanceNormalization':InstanceNormalization})
        model.load_weights('A:\アンケート\アンケートいいやつ\g_BA.h5')

        fftLen = 254
        win = np.hamming(fftLen)
        step = 62
        storage_spe = np.array([])
        comp = []
        ori = []
        log2500 = np.log(1000)
        A = 1000
        log4161536 = np.log(4161536)

        storage_spe = []
        a_b = []
        b_b = []
        start = time.time()

        while(True):
            xs = self.in_stream.read(self.chunk, exception_on_overflow=False)
            xs = np.frombuffer(xs,dtype="int16")

            self.title = self.fig.suptitle('{:.2f}'.format(time.time() - start) + 'sec')
            # self.realtime_line.set_ydata(xs)


            stft_data = self.stft(xs, win, step)
            abs_data = abs(stft_data[:, : fftLen // 2 + 1].T)

            Bs = []
            B = np.clip(abs_data, A, None)

            B = (np.log(B) - log2500) / (log4161536 - log2500) 
            a_b.extend(B.T)
            # a_b = np.concatenate([a_b,B], axis=1)
            
            if(len(a_b) > 80 * 3):
                a_b_ = [list(x) for x in zip(*a_b)]
                self.spe.set_data(a_b_)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events() 
                del a_b[0:5120]

            B = B[:, :, np.newaxis]
            Bs.append(B)
            Bs = np.array(Bs)

            predBs = model.predict(Bs.T)
            predBs = predBs.T

            backhalfspe = np.flipud(np.reshape(predBs, (128,80)))
            backhalfspe = backhalfspe[1:-1,:]
            predB = np.concatenate([predBs[0,:,:,0], backhalfspe])

            predB = np.exp(predB * (log4161536 - log2500) + log2500)
            predB[predB < A] = 1

            predB = np.pad(predB, [(0,0), (10,10)], 'constant')

            GLA = self.GriffinLim(predB.T, win, step, iterations=70)

            GLA = GLA[804:-804]



            # if(len(storage_spe) > 5120 * 5):
            spectrogram = self.stft(GLA, win, step)

            abs_data = abs(spectrogram[:, : fftLen // 2 + 1].T)

            Bs = []
            B = np.clip(abs_data, A, None)
            B = (np.log(B) - log2500) / (log4161536 - log2500) 


            # self.spe2.set_data(abs(spectrogram[:, : int(fftLen / 2) + 1].T))

            b_b.extend(B.T)
            if(len(b_b) > 80 * 3):
                b_b_ = [list(x) for x in zip(*b_b)]
                self.spe2.set_data(b_b_)
                del b_b[0:5120]

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

if __name__ == "__main__":
    a = RealTime_VoiceIO()
    a.update()

