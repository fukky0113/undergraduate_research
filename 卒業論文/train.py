import numpy as np
import sys
import os
import glob

import matplotlib
import matplotlib.pyplot as plt

from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json, Model
from keras.models import Model, Input
from keras.layers import Conv2D, UpSampling2D, Dropout, BatchNormalization, LeakyReLU, Reshape, Dense, Add, Concatenate, Flatten, Activation, Permute
from keras.utils import plot_model
import tensorflow as tf
from keras.backend import tensorflow_backend 

import scipy
from scipy.io import wavfile

from minibatch import MinibatchDiscrimination
from SpectralNormalizationKeras import ConvSN2D, DenseSN

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


class DataLoader():
    def __init__(self, img_res=(128, 64)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datasetting = DataSetting()
    
    def load_data(self,  epoch, batch_i, domain, batch_size=1, is_testing=False):
        MIN = 1000
        log4161536 = np.log(4161536)
        log1000 = np.log(MIN)
        path = glob.glob('データセットのパス/%s/*.wav' % domain)
        batch_images = np.random.choice(path, size=batch_size)
        imgs = []
        for num, img_path in enumerate(batch_images):
            rate, data = wavfile.read(img_path)
            spe = self.datasetting.stft(data, self.win, self.step)
            img = abs(spe[:, : int(self.fftLen / 2) + 1])
            img = np.clip(img, MIN, None)
            img = (np.log(img) - log1000) / (log4161536 - log1000) 
            img = img[:, :, np.newaxis]
            imgs.append(img)
            figori, axsori = plt.subplots(2,1)
            axsori[0].plot(data)
            axsori[1].imshow(abs(spe), cmap='gray', origin = "lower", aspect="auto")
            if(num == 0):
                figori.savefig("生成したデータの保存/images/originalA/" + str(epoch) + "_" + str(batch_i) +".png")
            else:
                figori.savefig("生成したデータの保存/images/originalB/" + str(epoch) + "_" + str(batch_i) +".png")
            plt.close()
        imgs = np.array(imgs)
        return imgs   

    def load_batch(self, batch_size=1, is_testing=False):
        MIN = 1000
        log4161536 = np.log(4161536)
        log1000 = np.log(MIN)
        self.fftLen = 254
        self.win = np.hamming(self.fftLen)
        self.step = 62
        path_A = glob.glob('データセットのパス/*.wav')
        path_B = glob.glob('データセットのパス/*.wav')
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        for i in range(self.n_batches - 1):    
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                rateA, dataA = wavfile.read(img_A)
                rateB, dataB = wavfile.read(img_B)
                dataA = np.ravel(dataA)
                dataB = np.ravel(dataB)
                speA = self.datasetting.stft(dataA, self.win, self.step)
                speB = self.datasetting.stft(dataB, self.win, self.step)
                img_A = abs(speA[:, : self.fftLen // 2 + 1])
                img_B = abs(speB[:, : self.fftLen // 2 + 1])           
                img_A = np.clip(img_A, MIN, None)
                img_A = (np.log(img_A) - log1000) / (log4161536 - log1000)              
                img_B = np.clip(img_B, MIN, None)
                img_B = (np.log(img_B) - log1000) / (log4161536 - log1000)            
                img_A = img_A[:, :, np.newaxis]
                img_B = img_B[:, :, np.newaxis]
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B

class Train():
    def __init__(self):
        self.datasetting = DataSetting()
        self.img_rows = 80
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.Dimg_shape = (self.img_rows, self.img_cols, self.channels)
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
        patch = int(self.img_cols / 2**4)
        self.disc_patch = (4, patch, 1)
        self.gf = 32
        self.df = 64
        self.lambda_cycle = 10.0
        self.lambda_id = 0.9 * self.lambda_cycle
        optimizer = Adam(0.0002, 0.5)
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        self.d_A.trainable = False
        self.d_B.trainable = False
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=optimizer)
        self.model_save(self.g_AB, "g_AB.json")
        self.model_save(self.g_BA, "g_BA.json")
        self.model_save(self.d_A, "d_A.json")
        self.model_save(self.d_B, "d_B.json")

    def model_save(self, model, name):
        j = model.to_json()
        open('重み保存パス/save/' + name, "w").write(j)

    def save_weight(self, epoch, batch):
        self.g_AB.save_weights('重み保存パス/save/g_AB.h5')
        self.g_BA.save_weights('重み保存パス/save/g_BA.h5')
        self.d_A.save_weights('重み保存パス/save/d_A.h5')
        self.d_B.save_weights('重み保存パス/save/d_B_z.h5')
        if(epoch % 200 == 0):
            self.g_AB.save_weights('重み保存パス/save/' + str(epoch) + '_' + str(batch) + "g_AB.h5")
            self.g_BA.save_weights('重み保存パス/save/' + str(epoch) + '_' + str(batch) + "g_BA.h5")
            self.d_A.save_weights('重み保存パス/save/' + str(epoch) + '_' + str(batch) + "d_A.h5")
            self.d_B.save_weights('重み保存パス/save/' + str(epoch) + '_' + str(batch) + "d_B_z.h5")

    def build_generator(self):
        d0 = Input(shape=self.img_shape)
        d1 = Conv2D(self.df - 1, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_uniform')(d0)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d = Reshape((80 , 128 * 1))(d0)
        d = Dense(units = 128,kernel_initializer='glorot_uniform')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d2 = Reshape((80, 128, 1))(d)
        c1 = Concatenate()([d1, d2])

        d1 = Conv2D(self.df - 1, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_uniform')(c1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d = Reshape((80 , 128 * 64))(c1)
        d = Dense(units = 128,kernel_initializer='glorot_uniform')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d2 = Reshape((80, 128, 1))(d)
        c1 = Concatenate()([d1, d2])

        d1 = Conv2D(self.df - 1, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_uniform')(c1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d = Reshape((80 , 128 * 64))(c1)
        d = Dense(units = 128,kernel_initializer='glorot_uniform')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d2 = Reshape((80, 128, 1))(d)
        c1 = Concatenate()([d1, d2])

        d1 = Conv2D(self.df - 1, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_uniform')(c1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d = Reshape((80 , 128 * 64))(c1)
        d = Dense(units = 128,kernel_initializer='glorot_uniform')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d2 = Reshape((80, 128, 1))(d)
        c1 = Concatenate()([d1, d2])

        d1 = Conv2D(self.df - 1, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_uniform')(c1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d = Reshape((80 , 128 * 64))(c1)
        d = Dense(units = 128,kernel_initializer='glorot_uniform')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d2 = Reshape((80, 128, 1))(d)
        c1 = Concatenate()([d1, d2])

        d1 = Conv2D(self.df - 1, kernel_size=3, strides=1, padding='same',kernel_initializer='glorot_uniform')(c1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d = Reshape((80 , 128 * 64))(c1)
        d = Dense(units = 128,kernel_initializer='glorot_uniform')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d2 = Reshape((80, 128, 1))(d)
        c1 = Concatenate()([d1, d2])

        output_img = Conv2D(self.channels, kernel_size=3,strides=1, padding='same', activation='sigmoid',kernel_initializer='glorot_uniform')(c1)
        model = Model(d0, output_img)
        model.summary()
        plot_model(model, to_file='gmodel.png', show_shapes=True)
        return model
                        
    def build_discriminator(self):
        img = Input(shape=self.Dimg_shape)

        d1 = ConvSN2D(self.df, 4, strides=2, padding='same',kernel_initializer='glorot_uniform')(img)
        d2 = LeakyReLU(alpha=0.2)(d1)

        d2 = ConvSN2D(self.df * 2, 4, strides=2, padding='same',kernel_initializer='glorot_uniform')(d2)
        d3 = LeakyReLU(alpha=0.2)(d2)

        d3 = ConvSN2D(self.df * 4, 4, strides=2, padding='same',kernel_initializer='glorot_uniform')(d3)
        d4 = LeakyReLU(alpha=0.2)(d3)

        d4 = ConvSN2D(self.df * 8, 4, strides=2, padding='same',kernel_initializer='glorot_uniform')(d4)
        d6 = LeakyReLU(alpha=0.2)(d4)

        d5 = ConvSN2D(self.df * 16, 4, strides=2, padding='same',kernel_initializer='glorot_uniform')(d6)
        d6 = LeakyReLU(alpha=0.2)(d5)

        d6 = Flatten()(d6)
        d7 = MinibatchDiscrimination(12, 3)(d6)
        d7 = Reshape((3, 4, 1025))(d7)

        validity = ConvSN2D(1, 4, strides=1, padding='same',kernel_initializer='glorot_uniform')(d7)
        model = Model(img, validity)       
        model.summary()
        plot_model(model, to_file='dmodel.png', show_shapes=True)
        return model 

    def sample_images(self, epoch, batch_i):
        r, c = 2, 3
        MIN = 1000
        log4161536 = np.log(4161536)
        log1000 = np.log(MIN)
        imgs_A = self.data_loader.load_data(epoch, batch_i, domain="データセットのパス", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(epoch, batch_i, domain="データセットのパス", batch_size=1, is_testing=True)
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
        gen_imgs = np.concatenate([imgs_A.T, fake_B.T, reconstr_A.T, imgs_B.T, fake_A.T, reconstr_B.T])
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray', origin = "lower", aspect="auto")
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("生成したデータの保存/images/spectrogram/%d_%d.png" % (epoch, batch_i))
        plt.close()

        win = self.data_loader.win
        step = self.data_loader.step
        fftlen = self.data_loader.fftLen

        backhalfspe = np.flipud(gen_imgs[0, :, :, 0])
        backhalfspe = backhalfspe[1:-1,:]
        originalA = np.concatenate([gen_imgs[0, :, :, 0], backhalfspe])
        
        backhalfspe = np.flipud(gen_imgs[1, :, :, 0])
        backhalfspe = backhalfspe[1:-1,:]
        fakeA = np.concatenate([gen_imgs[1, :, :, 0], backhalfspe])

        backhalfspe = np.flipud(gen_imgs[2, :, :, 0])
        backhalfspe = backhalfspe[1:-1,:]
        restorespeA = np.concatenate([gen_imgs[2, :, :, 0], backhalfspe])

        backhalfspe = np.flipud(gen_imgs[3, :, :, 0])
        backhalfspe = backhalfspe[1:-1,:]
        originalB = np.concatenate([gen_imgs[3, :, :, 0], backhalfspe])

        backhalfspe = np.flipud(gen_imgs[4, :, :, 0])
        backhalfspe = backhalfspe[1:-1,:]
        fakeB = np.concatenate([gen_imgs[4, :, :, 0], backhalfspe])
        
        backhalfspe = np.flipud(gen_imgs[5, :, :, 0])
        backhalfspe = backhalfspe[1:-1,:]
        restorespeB = np.concatenate([gen_imgs[5, :, :, 0], backhalfspe])

        originalB = np.exp(originalB * (log4161536 - log1000) + log1000)
        fakeB = np.exp(fakeB * (log4161536 - log1000) + log1000)
        restorespeB = np.exp(restorespeB * (log4161536 - log1000) + log1000)
        restorespeA = np.exp(restorespeA * (log4161536 - log1000) + log1000)
        fakeA = np.exp(fakeA * (log4161536 - log1000) + log1000)
        originalA = np.exp(originalA * (log4161536 - log1000) + log1000)

        originalB[originalB < MIN] = 1
        fakeB[fakeB < MIN] = 1
        restorespeB[restorespeB < MIN] = 1
        restorespeA[restorespeA < MIN] = 1
        fakeA[fakeA < MIN] = 1
        originalA[originalA < MIN] = 1

        oriA = self.datasetting.GriffinLim(originalA.T, win, step)
        oriB = self.datasetting.GriffinLim(originalB.T, win, step)

        fakeA = self.datasetting.GriffinLim(fakeA.T, win, step)
        fakeB = self.datasetting.GriffinLim(fakeB.T, win, step)

        reA = self.datasetting.GriffinLim(restorespeA.T, win, step)
        reB = self.datasetting.GriffinLim(restorespeB.T, win, step)

        wavfile.write('生成したデータの保存/GLA_A/%d_%d.wav' % (epoch, batch_i) , 16000, (oriA).astype(np.int16))
        wavfile.write('生成したデータの保存/GLA_B/%d_%d.wav' % (epoch, batch_i) , 16000, (oriB).astype(np.int16))
        
        wavfile.write('生成したデータの保存/AtoB/%d_%d.wav' % (epoch, batch_i) , 16000, (fakeA).astype(np.int16))
        wavfile.write('生成したデータの保存/BtoA/%d_%d.wav' % (epoch, batch_i) , 16000, (fakeB).astype(np.int16))
        
        wavfile.write('生成したデータの保存/AtoBtoA/%d_%d.wav' % (epoch, batch_i) , 16000, (reA).astype(np.int16))
        wavfile.write('生成したデータの保存/BtoAtoB/%d_%d.wav' % (epoch, batch_i) , 16000, (reB).astype(np.int16))
 
        figA, axsA = plt.subplots(3,1)
        axsA[0].plot((oriA).astype(np.int16))
        axsA[1].plot((fakeA).astype(np.int16))
        axsA[2].plot((reA).astype(np.int16))

        axsA[0].set_title('oriA')
        axsA[1].set_title('fakeA')
        axsA[2].set_title('reA')

        figB, axsB = plt.subplots(3,1)
        axsB[0].plot((oriB).astype(np.int16))
        axsB[1].plot((fakeB).astype(np.int16))
        axsB[2].plot((reB).astype(np.int16))

        axsB[0].set_title('oriB')
        axsB[1].set_title('fakeB')
        axsB[2].set_title('reB')

        figA.savefig("生成したデータの保存/images/A/%d_%d.png" % (epoch, batch_i))
        figB.savefig("生成したデータの保存/images/B/%d_%d.png" % (epoch, batch_i))
        plt.close()
        plt.close()

        if (True):
            rateA, dataA = wavfile.read('テストデータのパス/full_voice/.wav')
            rateB, dataB = wavfile.read('テストデータのパス/full_voice/.wav')
            dataA = np.ravel(dataA)
            dataB = np.ravel(dataB)
            dataA = np.pad(dataA, [1280,0], 'mean')
            dataB = np.pad(dataB, [1280,0], 'mean')
            a = []
            b = []
            for i in range((len(dataA)//2560) - 1):
                imgs = []
                cutdata = dataA[2560 * i: 2560 * (i + 2)]
                spe = self.datasetting.stft(cutdata, win, step)
                img = abs(spe[:, : int(fftlen / 2) + 1])
                img = np.clip(img, MIN, None)
                img = (np.log(img) - log1000) / (log4161536 - log1000) 
                img = img[:, :, np.newaxis]
                imgs.append(img)
                imgs = np.array(imgs)
                fake = self.g_AB.predict(imgs)
                fake = fake.T
                backhalfspe = np.flipud(fake[0,:,:,0])
                backhalfspe = backhalfspe[1:-1,:]
                restorespeA = np.concatenate([fake[0,:,:,0], backhalfspe])
                restorespeA = np.exp(restorespeA * (log4161536 - log1000) + log1000)
                restorespeA[restorespeA < MIN] = 1
                fakeA = self.datasetting.GriffinLim(restorespeA.T, win, step)
                fakeA = fakeA[1280:3840]
                a.extend(fakeA)

            for i in range(len(dataB)//2560 - 1):
                imgsB = []
                cutdataB = dataB[2560 * i: 2560 * (i + 2)]
                speB = self.datasetting.stft(cutdataB, win, step)
                imgB = abs(speB[:, : int(fftlen / 2) + 1])
                imgB = np.clip(imgB, MIN, None)
                imgB = (np.log(imgB) - log1000) / (log4161536 - log1000) 
                imgB = imgB[:, :, np.newaxis]
                imgsB.append(imgB)
                imgsB = np.array(imgsB)
                fakeB = self.g_BA.predict(imgsB)
                fakeB = fakeB.T
                backhalfspeB = np.flipud(fakeB[0,:,:,0])
                backhalfspeB = backhalfspeB[1:-1,:]
                restorespeB = np.concatenate([fakeB[0,:,:,0], backhalfspeB])
                restorespeB = np.exp(restorespeB * (log4161536 - log1000) + log1000)
                restorespeB[restorespeB < MIN] = 1
                fakeBB = self.datasetting.GriffinLim(restorespeB.T, win, step)
                fakeBB = fakeBB[1280:3840]
                b.extend(fakeBB)
            a = np.array(a)
            b = np.array(b)
            wavfile.write('生成したデータの保存/full/A/%d_%d.wav' % (epoch, batch_i) , 16000, (a).astype(np.int16))
            wavfile.write('生成したデータの保存/full/B/%d_%d.wav' % (epoch, batch_i) , 16000, (b).astype(np.int16))
        plt.close()

    def train(self, epochs, batch_size, sample_interval=50):
        valid = np.ones((batch_size,) + (3, 4, 1))
        fake = np.zeros((batch_size,) + (3, 4, 1))
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                dA_loss_real = self.d_A.train_on_batch(D_imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(D_fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(D_imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(D_fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

                log = ('epoch:{:4}, batch:{:2}, D: loss:{:3.4f}, acc:{:3.4f}, G: loss:{:3.4f}, model_1_loss:{:3.4f}, model_2_loss:{:3.4f}, model_4_loss:{:3.4f}, model_3_loss:{:3.4f}, model_4_loss:{:3.4f}, model_3_loss:{:3.4f}').format(epoch, batch_i, d_loss[0], d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6])

                with open('/home/fukiyama/Lab/CycleGAN/log.txt', 'a') as f:
                    f.write(log + '\n')
                
                if (epoch % 2 == 0 and batch_i == 0):
                    self.sample_images(epoch, batch_i)
                    self.save_weight(epoch, batch_i)
                    print(log)


class Train(Train):
    @staticmethod
    def conv2d(layer_input, filters, f_size=4, f_strides=2, normalization=True, dropout_rate=0):
        d = Conv2D(filters, kernel_size=f_size, strides=f_strides, padding='same')(layer_input)
        c = LeakyReLU(alpha=0.2)(d)
        if normalization:
            c = BatchNormalization()(c)
        return c

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, f_strides=1, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=f_strides, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization()(u)
        if(skip_input != None):
            u = Concatenate()([u, skip_input])
        return u

    @staticmethod
    def _shortcut(input, residual):
        n_filters = residual._keras_shape[3]
        shortcut = Conv2D(n_filters, (1, 1), strides=(1,1), padding='valid')(input)
        return Add()([shortcut, residual])

    @staticmethod
    def _resblock(n_filters, strides=(1,1)):
        def f(input):
            x = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(input)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(x)
            return Train._shortcut(input, x)
        return f

if __name__ == '__main__':
   
    config = tf.compat.v1.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    t = Train()
    t.train(epochs = 4000, batch_size = 16, sample_interval = 2)

 