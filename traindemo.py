import numpy as np
import sys
import os
import glob

import matplotlib
import matplotlib.pyplot as plt

from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json, Model

from keras.models import Model, Input
from keras.layers import Conv2D, UpSampling2D, Dropout, BatchNormalization, LeakyReLU, Reshape, Dense, Add, Concatenate, Flatten, Activation
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from keras.utils import plot_model

from distutils.dir_util import copy_tree

""" GPU������΁AGPU�������̎g�p�ʂ��w�� """
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend 

import time
import scipy

from scipy.io import wavfile


# d = str(device_lib.list_local_devices())
# if "GPU" in d:
#     # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))  # ���A���^�C���ŕK�v�ȕ������m��
#     config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))  # �󂫃������̉������g�������w��
#     # config = tf.ConfigProto(device_count={"GPU": 0})  # GPU���g��Ȃ�(GPU���������ǂ����Ă�����Ȃ��Ƃ�)
#     session = tf.compat.v1.Session(config=config)
#     tensorflow_backend.set_session(session)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='0')) 
#     session = tf.Session(config=config)
#     tensorflow_backend.set_session(session)
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# gpuConfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list="0"))
# sess = tf.Session(config=gpuConfig)
# tensorflow_backend.set_session(sess)

class DataSetting():
    def __init__(self):
        pass

    def cut5120(self, data):
        datalen = len(data)
        random = np.random.randint(0, datalen - 5120)
        return data[random : random + 5120]


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


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 64)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.datasetting = DataSetting()
    
    def load_data(self,  epoch, batch_i, domain, batch_size=1, is_testing=False):
        path = glob.glob('C:/Lab/demo/CycleGanDemoVoiceVSN/%s/data/N%s2/*.wav' % (self.dataset_name, domain))
        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            rate, data = wavfile.read(img_path)
            data = np.ravel(data)
            data = self.datasetting.cut5120(data)
            spe = self.datasetting.stft(data, self.win, self.step)
            # spe = np.array(spe, dtype=np.float32).copy()
            # spe /= 32768.0
            img = abs(spe[:, : int(self.fftLen / 2) + 1].T)

            img = np.clip(img, 2500, None)
            img = (np.log(img) - np.log(2500)) / (np.log(4161536) - np.log(2500)) 

            img = img[:, :, np.newaxis]
            imgs.append(img)

            figori, axsori = plt.subplots(2,1)
            axsori[0].plot(data)
            axsori[1].imshow(abs(spe).T, cmap='gray', origin = "lower", aspect="auto")
            figori.savefig("C:/Lab/demo/CycleGanDemoVoiceVSN/result/images/original" + domain + "/" + str(epoch) + "_" + str(batch_i) +".png")
            plt.close()


        imgs = np.array(imgs)
        return imgs   

    def load_batch(self, batch_size=1, is_testing=False):
        self.fftLen = 254
        self.win = np.hamming(self.fftLen)
        self.step = 78

        path_A = glob.glob('C:/Lab/demo/CycleGanDemoVoiceVSN/%s/data/%s/*.wav' % (self.dataset_name, 'NA-human2'))
        path_B = glob.glob('C:/Lab/demo/CycleGanDemoVoiceVSN/%s/data/%s/*.wav' % (self.dataset_name, 'NB2-myvoice3'))

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
                dataA = self.datasetting.cut5120(dataA)
                dataB = self.datasetting.cut5120(dataB)
                speA = self.datasetting.stft(dataA, self.win, self.step)
                speB = self.datasetting.stft(dataB, self.win, self.step)
                # speA = np.array(speA, dtype=np.float32).copy()
                # speB = np.array(speB, dtype=np.float32).copy()
                # speA /= 32768.0
                # speB /= 32768.0
                img_A = abs(speA[:, : self.fftLen // 2 + 1].T)
                img_B = abs(speB[:, : self.fftLen // 2 + 1].T)

                
                img_A = np.clip(img_A, 2500, None)
                img_A = (np.log(img_A) - np.log(2500)) / (np.log(4161536) - np.log(2500)) 
                
                img_B = np.clip(img_B, 2500, None)
                img_B = (np.log(img_B) - np.log(2500)) / (np.log(4161536) - np.log(2500)) 
                # img_A = abs(speA.T)
                # img_B = abs(speB.T)
                    
                # img_A = img_A.tolist()
                # img_B = img_B.tolist()    
                
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

        self.img_rows = 128
        self.img_cols = 64

        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'materials'
        # Use the DataLoader object to import a preprocessed dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))

        patch = int(self.img_rows / 2**4)
        #(8,8,1)
        self.disc_patch = (patch, 4, 1)

        self.gf = 32
        self.df = 64

        self.lambda_cycle = 10.0
        self.lambda_id = 0.9 * self.lambda_cycle

        # �I�v�e�B�}�C�U=�œK���A���S���Y��
        optimizer = Adam(0.0002, 0.5)

        # Da��Db���쐬���A�R���p�C��
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # self.d_B.summary()

        # 2�̐�������쐬
        # Gab��Gba���C���X�^���X��
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        # self.g_BA.summary()

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # A��B'��A'' B��A'��B'' �T�C�N����ѐ�����
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # A��A' and B��B' ���ꐫ��������
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # generator �݂̂��w�K���邽�߂ɁAdiscriminator�̊w�K���~�߂�
        self.d_A.trainable = False
        self.d_B.trainable = False

        # generator���쐬�����U����discriminator��
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
        open('C:/Lab/demo/CycleGanDemoVoiceVSN/save/' + name, "w").write(j)

    def save_weight(self, epoch, batch):
        self.g_AB.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/g_AB.h5')
        self.g_BA.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/g_BA.h5')
        self.d_A.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/d_A.h5')
        self.d_B.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/d_B_z.h5')
        if(epoch % 100 == 0):
            self.g_AB.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/' + str(epoch) + '_' + str(batch) + "g_AB.h5")
            self.g_BA.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/' + str(epoch) + '_' + str(batch) + "g_BA.h5")
            self.d_A.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/' + str(epoch) + '_' + str(batch) + "d_A.h5")
            self.d_B.save_weights('C:/Lab/demo/CycleGanDemoVoiceVSN/save/' + str(epoch) + '_' + str(batch) + "d_B_z.h5")

    def build_generator(self):
        d0 = Input(shape=self.img_shape)

        d1 = self.conv2d(d0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)

        d3 = self._resblock(self.gf * 4)(d3)
        d3 = self._resblock(self.gf * 4)(d3)
        d3 = self._resblock(self.gf * 4)(d3)
        d3 = self._resblock(self.gf * 4)(d3)
        d3 = self._resblock(self.gf * 4)(d3)

        u1 = self.deconv2d(d3, None, self.gf * 4)
        u2 = self.deconv2d(u1, None, self.gf * 2)
        u3 = self.deconv2d(u2, None, self.gf)

        # u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4,strides=1, padding='same', activation='sigmoid')(u3)

        # d = Flatten()(d0)
        # d = Dense(units = 128 * 64 // 2)(d)
        # d = Dropout(0.5)(d)
        # d = LeakyReLU(alpha = 0.2)(d)
        # d = Dense(units = 128 * 64 // 4)(d)
        # d = Dropout(0.5)(d)
        # d = LeakyReLU(alpha = 0.2)(d)
        # d = Dense(units = 128 * 64, activation = "sigmoid")(d)
        # d = Reshape((128, 64, 1))(d)

        # output_img = Add()([output_img, d])
        model = Model(d0, output_img)

        model.summary()
        
        plot_model(model, to_file='gmodel.png', show_shapes=True)

        return model



        # d0 = Input(shape=self.img_shape)

        # d1 = self.conv2d(d0, self.gf)
        # d2 = self.conv2d(d1, self.gf * 2)
        # d3 = self.conv2d(d2, self.gf * 4)
        # d4 = self.conv2d(d3, self.gf * 8)

        # u1 = self.deconv2d(d4, None, self.gf * 4)
        # u2 = self.deconv2d(u1, None, self.gf * 2)
        # u3 = self.deconv2d(u2, None, self.gf)
 
        # # u1 = self.deconv2d(d4, d3, self.gf * 4)
        # # u2 = self.deconv2d(u1, d2, self.gf * 2)
        # # u3 = self.deconv2d(u2, d1, self.gf)

        # u4 = UpSampling2D(size=2)(u3)
        # output_img = Conv2D(self.channels, kernel_size=4,strides=1, padding='same', activation='tanh')(u4)

        # # d = Flatten()(d0)
        # # d = Dense(units = 64 * 80 // 2)(d)
        # # d = Dropout(0.5)(d)
        # # d = LeakyReLU(alpha = 0.2)(d)
        # # d = Dense(units = 64 * 80 // 4)(d)
        # # d = Dropout(0.5)(d)
        # # d = LeakyReLU(alpha = 0.2)(d)
        # # d = Dense(units = 64 * 80, activation = "sigmoid")(d)
        # # d = Reshape((64, 80, 1))(d)
        # # output = Add()([output_img, d])

        # model = Model(d0, output_img)
        # # model.summary()
        # # plot_model(model, to_file='gmodel.png', show_shapes=True)

        # return model

    def build_discriminator(self):
        img = Input(shape=self.img_shape)

        d1 = self.conv2d(img, self.df, normalization=False)
        d2 = self.conv2d(d1, self.df * 2, dropout_rate=0.5)
        d3 = self.conv2d(d2, self.df * 4, dropout_rate=0.5)
        d4 = self.conv2d(d3, self.df * 8, dropout_rate=0.5)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        # c = LeakyReLU(0.2)(d4)
        # c = Flatten()(c)
        # c = Dense(units=128)(c)
        # c = LeakyReLU(0.2)(c)
        # output = Dense(units=1, activation="sigmoid")(c)


        model = Model(img, validity)
        # model = Model(img, output)
        
        # model.summary()
        # plot_model(model, to_file='dmodel_VSN.png', show_shapes=True)

        return model 

    def sample_images(self, epoch, batch_i):
        r, c = 2, 3
        imgs_A = self.data_loader.load_data(epoch, batch_i, domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(epoch, batch_i, domain="B", batch_size=1, is_testing=True)
        
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)


        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # 意味不明
        # gen_imgs = 0.5 * gen_imgs + 0.5
        # np.delete(gen_imgs, 1, 0)
        
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray', origin = "lower", aspect="auto")
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("C:/Lab/demo/CycleGanDemoVoiceVSN/result/images/spectrogram/%d_%d.png" % (epoch, batch_i))
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

        # print(restorespeB.shape)

        originalB = np.exp(originalB * (np.log(4161536) - np.log(2500)) + np.log(2500))
        fakeB = np.exp(fakeB * (np.log(4161536) - np.log(2500)) + np.log(2500))
        restorespeB = np.exp(restorespeB * (np.log(4161536) - np.log(2500)) + np.log(2500))
        restorespeA = np.exp(restorespeA * (np.log(4161536) - np.log(2500)) + np.log(2500))
        fakeA = np.exp(fakeA * (np.log(4161536) - np.log(2500)) + np.log(2500))
        originalA = np.exp(originalA * (np.log(4161536) - np.log(2500)) + np.log(2500))

        originalB[originalB < 2500] = 1
        fakeB[fakeB < 2500] = 1
        restorespeB[restorespeB < 2500] = 1
        restorespeA[restorespeA < 2500] = 1
        fakeA[fakeA < 2500] = 1
        originalA[originalA < 2500] = 1

        oriA = self.datasetting.GriffinLim(originalA.T, win, step)
        oriB = self.datasetting.GriffinLim(originalB.T, win, step)

        fakeA = self.datasetting.GriffinLim(fakeA.T, win, step)
        fakeB = self.datasetting.GriffinLim(fakeB.T, win, step)

        reA = self.datasetting.GriffinLim(restorespeA.T, win, step)
        reB = self.datasetting.GriffinLim(restorespeB.T, win, step)

        wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/GLA_A/%d_%d.wav' % (epoch, batch_i) , 16000, (oriA).astype(np.int16))
        wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/GLA_B/%d_%d.wav' % (epoch, batch_i) , 16000, (oriB).astype(np.int16))
        
        wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/AtoB/%d_%d.wav' % (epoch, batch_i) , 16000, (fakeA).astype(np.int16))
        wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/BtoA/%d_%d.wav' % (epoch, batch_i) , 16000, (fakeB).astype(np.int16))
        
        wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/AtoBtoA/%d_%d.wav' % (epoch, batch_i) , 16000, (reA).astype(np.int16))
        wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/BtoAtoB/%d_%d.wav' % (epoch, batch_i) , 16000, (reB).astype(np.int16))
 
        figA, axsA = plt.subplots(3,1)
        axsA[0].plot((oriA).astype(np.int16))
        axsA[1].plot((fakeA).astype(np.int16))
        axsA[2].plot((reA).astype(np.int16))

        axsA[0].set_title('oriA')
        axsA[1].set_title('fakeA')
        axsA[2].set_title('reA')
        # axsA[0].set_ylim(-1,1)
        # axsA[1].set_ylim(-1,1)
        # axsA[2].set_ylim(-1,1)

        figB, axsB = plt.subplots(3,1)
        axsB[0].plot((oriB).astype(np.int16))
        axsB[1].plot((fakeB).astype(np.int16))
        axsB[2].plot((reB).astype(np.int16))

        axsB[0].set_title('oriB')
        axsB[1].set_title('fakeB')
        axsB[2].set_title('reB')
        # axsB[0].set_ylim(-1,1)
        # axsB[1].set_ylim(-1,1)
        # axsB[2].set_ylim(-1,1)

        figA.savefig("C:/Lab/demo/CycleGanDemoVoiceVSN/result/images/A/%d_%d.png" % (epoch, batch_i))
        figB.savefig("C:/Lab/demo/CycleGanDemoVoiceVSN/result/images/B/%d_%d.png" % (epoch, batch_i))
        plt.close()
        plt.close()

        # plt.show()
        if (epoch % 1 == 0 and batch_i == 133):
            rateA, dataA = wavfile.read('C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NA-human/UT-PARAPHRASE-sent060-phrase2.wav')
            # rateA, dataA = wavfile.read('C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NA2/あなたが嬉しいと、わたしも嬉しいです.wav')
            # rateB, dataB = wavfile.read('./demo/CycleGanDemoVoiceVSN/materials/data/NB2/あなたが嬉しいと、わたしも嬉しいです.wav')
            rateB, dataB = wavfile.read('C:/Lab/demo/CycleGanDemoVoiceVSN/materials/data/NB2-myvoice2/夕日は大きなトンネルになった.wav')
            dataA = np.ravel(dataA)
            dataB = np.ravel(dataB)

            dataA = np.pad(dataA, [1280,0], 'mean')
            dataB = np.pad(dataB, [1280,0], 'mean')
            
            # imgs_A = self.data_loader.load_data(epoch, batch_i, domain="A", batch_size=1, is_testing=True)
            a = []
            b = []
            for i in range((len(dataA)//2560) - 1):
                imgs = []
                cutdata = dataA[2560 * i: 2560 * (i + 2)]

                spe = self.datasetting.stft(cutdata, win, step)
                # spe = np.array(spe, dtype=np.float32).copy()
                # spe /= 32768.0
                img = abs(spe[:, : int(fftlen / 2) + 1].T)
            
                img = np.clip(img, 2500, None)
                img = (np.log(img) - np.log(2500)) / (np.log(4161536) - np.log(2500)) 

                img = img[:, :, np.newaxis]
                imgs.append(img)
                imgs = np.array(imgs)

                fake = self.g_AB.predict(imgs)
                backhalfspe = np.flipud(fake[0,:,:,0])
                backhalfspe = backhalfspe[1:-1,:]
                restorespeA = np.concatenate([fake[0,:,:,0], backhalfspe])
                
                restorespeA = np.exp(restorespeA * (np.log(4161536) - np.log(2500)) + np.log(2500))
                restorespeA[restorespeA < 2500] = 1
                fakeA = self.datasetting.GriffinLim(restorespeA.T, win, step)

                # if (i != 0):
                fakeA = fakeA[1280:3840]
                # else:
                #     fakeA = fakeA[0:3840]

                a.extend(fakeA)

            for i in range(len(dataB)//2560 - 1):
                imgsB = []
                cutdataB = dataB[2560 * i: 2560 * (i + 2)]

                speB = self.datasetting.stft(cutdataB, win, step)
                # spe = np.array(spe, dtype=np.float32).copy()
                # spe /= 32768.0
                imgB = abs(speB[:, : int(fftlen / 2) + 1].T)
            
                imgB = np.clip(imgB, 2500, None)
                imgB = (np.log(imgB) - np.log(2500)) / (np.log(4161536) - np.log(2500)) 

                imgB = imgB[:, :, np.newaxis]
                imgsB.append(imgB)
                imgsB = np.array(imgsB)

                fakeB = self.g_BA.predict(imgsB)
                backhalfspeB = np.flipud(fakeB[0,:,:,0])
                backhalfspeB = backhalfspeB[1:-1,:]
                restorespeB = np.concatenate([fakeB[0,:,:,0], backhalfspeB])

                restorespeB = np.exp(restorespeB * (np.log(4161536) - np.log(2500)) + np.log(2500))
                restorespeB[restorespeB < 2500] = 1

                fakeBB = self.datasetting.GriffinLim(restorespeB.T, win, step)

                # if (i != 0):
                fakeBB = fakeBB[1280:3840]
                # else:
                #     fakeA = fakeA[0:3840]

                b.extend(fakeBB)
            
            a = np.array(a)
            b = np.array(b)
            
            wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/full/A/%d_%d.wav' % (epoch, batch_i) , 16000, (a).astype(np.int16))
            wavfile.write('C:/Lab/demo/CycleGanDemoVoiceVSN/result/full/B/%d_%d.wav' % (epoch, batch_i) , 16000, (b).astype(np.int16))
        
        plt.close()


    def train(self, epochs, batch_size, sample_interval=50):
        # shape:(batch_size, 8, 8, 1)
        
        #https://qiita.com/underfitting/items/a0cbb035568dea33b2d7
        #普通はReal=1、Fake=0とするが、Real=0.7~1.2、Fake=0.0〜0.3からランダムにサンプルする。(Label Smoothing) Salimans et. al. 2016
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        #Discriminator変更したときに変更した
        # valid = np.ones((batch_size,) + (1,))
        # fake = np.zeros((batch_size,) + (1,))

        # valid = np.random.uniform(0.7, 1.0, ((batch_size,) + self.disc_patch))
        # fake = np.random.uniform(0.0, 0.3, ((batch_size,) + self.disc_patch))

        # valid = np.ones((batch_size,) + self.disc_patch) * round(np.random.uniform(0.7, 1.2), 1)
        # fake = np.ones((batch_size,) + self.disc_patch)* round(np.random.uniform(0.0, 0.3), 1)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                
                # discriminator
                # print(imgs_A.shsape)
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                # print('dA_loss_real:' + str(dA_loss_real[0]) + 'dA_loss_fake:' + str(dA_loss_fake[0]))
                # print('log(dA_loss_real):' + str(np.log(dA_loss_real[0])) + 'log(dA_loss_fake):' + str(np.log(1 - dA_loss_fake[0])))

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # discriminator �̍��vloss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # generator
                # inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id]
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])


                log = ('epoch:{:4}, batch:{:2}, D: loss:{:3.4f}, acc: {:3.4f}, G: loss:{:3.4f}, model_1_loss:{:3.4f}, model_2_loss:{:3.4f}, model_4_loss:{:3.4f}, model_3_loss:{:3.4f}, model_4_loss:{:3.4f}, model_3_loss:{:3.4f}').format(epoch, batch_i, d_loss[0], d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6])
                # print(log)
                # f = open('./demo/CycleGanDemovoice/log.txt', 'a').write(log+ '\n')
                
                with open('C:/Lab/demo/CycleGanDemoVoiceVSN/log.txt', 'a') as f:
                    f.write(log + '\n')
                
                # Dislog = ('epoch:{:4}, batch:{:2}, log(dA_loss_real:{:3.4f}, log(dA_loss_fake:{:3.4f}, log(dB_loss_real:{:3.4f}, log(dB_loss_fake:{:3.4f}').format(epoch, batch_i,np.log(dA_loss_real[0]), np.log(1 - dA_loss_fake[0]), np.log(dB_loss_real[0]), np.log(1 - dB_loss_fake[0]))
                # with open('C:/Lab/demo/CycleGanDemoVoiceVSN/Discriminator_log.txt', 'a') as f:
                #     f.write(Dislog + '\n')
                
                # Genlog = ('epoch:{:4}, batch:{:2}, log(dA_loss_fake:{:3.4f}, log(dB_loss_fake:{:3.4f}').format(epoch, batch_i, np.log(1 - g_loss[1]), np.log(1 - g_loss[2]))
                # with open('C:/Lab/demo/CycleGanDemoVoiceVSN/Generator_log.txt', 'a') as f:
                #     f.write(Genlog + '\n')
                
                if (batch_i == sample_interval):

                    # print(self.d_A.predict(fake_A))

                    self.sample_images(epoch, batch_i)
                    self.save_weight(epoch, batch_i)
                    print(log)

                    # print('dA_loss_real:' + str(dA_loss_real[0]) + '  dA_loss_fake:' + str(dA_loss_fake[0]))
                    # print('log(dA_loss_real):' + str(np.log(dA_loss_real[0])) + '  log(dA_loss_fake):' + str(np.log(1 - dA_loss_fake[0])))

                    # print('dA_loss_real:' + str(dB_loss_real[0]) + '  dA_loss_fake:' + str(dB_loss_fake[0]))
                    # print('log(dA_loss_real):' + str(np.log(dB_loss_real[0])) + '  log(dA_loss_fake):' + str(np.log(1 - dB_loss_fake[0])))


                # print(self.combined.metrics_names)

class Train(Train):
    @staticmethod
    def conv2d(layer_input, filters, f_size=4, normalization=True, dropout_rate=0):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        c = LeakyReLU(alpha=0.2)(d)
        if normalization:
            c = BatchNormalization()(c)
        return c

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                    padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization()(u)
        if(skip_input != None):
            u = Concatenate()([u, skip_input])
        return u

    @staticmethod
    def _shortcut(input, residual):
        # _keras_shape[3] channel num
        n_filters = residual._keras_shape[3]

        # inputsとresidualでチェンネルの数が違う可能性あり、そのままだと足せないので1x1convでresidual側のフィルタに合わせている
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
    
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    t = Train()
    t.train(epochs = 50000, batch_size = 8, sample_interval = 133)

# path = os.getcwd()
# print(path)

# path_A = glob.glob('./demo/CycleGanDemo/%s/%sA/*' % ('apple2orange', 'train'))
# path_B = glob.glob('./demo/CycleGanDemo/%s/%sB/*' % ('apple2orange', 'train'))

# print(len(path_A), len(path_B))
 