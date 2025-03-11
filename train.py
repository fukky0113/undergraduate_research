import numpy as np
import sys
import os
import glob

import matplotlib
import matplotlib.pyplot as plt

from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json, Model

from keras.models import Model, Input
from keras.layers import Conv2D, UpSampling2D, Dropout, BatchNormalization, LeakyReLU, Reshape, Dense, Add, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from distutils.dir_util import copy_tree

""" GPUがあれば、GPUメモリの使用量を指定 """
import tensorflow as tf
from tensorflow.python.client import device_lib
# from keras.backend import tensorflow_backend 
from keras import backend as K

import time
import scipy
from scipy.io import wavfile


# d = str(device_lib.list_local_devices())
# if "GPU" in d:
#     # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))  # リアルタイムで必要な分だけ確保
#     config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))  # 空きメモリの何割を使うかを指定
#     # config = tf.ConfigProto(device_count={"GPU": 0})  # GPUを使わない(GPUメモリがどうしても足らないとき)
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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# config = tf.ConfigProto(
#     gpu_options=tf.GPUOptions(
#         per_process_gpu_memory_fraction=0.8, # 最大値の80%まで
#         allow_growth=True # True->必要になったら確保, False->全部
#     )
# )
sess = sess = tf.Session(config=config)

class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 320)):
        self.dataset_name = dataset_name
        self.img_res = img_res
    
    #短時間フーリエ変換(データ,長さ?フレーム?用語の名前が分からん)
    def stft(self, x, win, step):
        l = len(x)
        N = len(win)
        M = int(np.ceil(float(l - N + step)/ step))

        new_x = np.zeros(N + ((M - 1) * step))#, dtype = np.float64)
        # new_x = [0] * (N + ((M - 1) * step))
        new_x[: l] = x 
        X = np.zeros([M, N])#, dtype = np.complex64)
        for m in range(M):
            start = int(step) * m
            X[m, :] = np.fft.fft(new_x[start : start + N]* win)

        return X

    def load_data(self, domain, batch_size=1, is_testing=False):
        path = glob.glob('./demo/CycleGanDemovoice/%s/data/New_data_%s/*.wav' % (self.dataset_name, domain))
        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            data = wavfile.read(img_path)
            data = np.ravel(data)
            spe = stft(data, win, step)
            spe = np.array(spe, dtype=np.float32).copy()
            spe /= 32768.0
            img = abs(spe[:, : int(fftLen / 2) + 1].T)
            imgs.append(img)
        return imgs   

    def load_batch(self, batch_size=1, is_testing=False):
        fftLen = 1024 // 2
        win = np.hamming(fftLen)
        step = fftLen // 4

        data_type = "data_"
        path_A = glob.glob('./demo/CycleGanDemovoice/%s/data/New_%sA/*.wav' % (self.dataset_name, data_type))
        path_B = glob.glob('./demo/CycleGanDemovoice/%s/data/New_%sB/*.wav' % (self.dataset_name, data_type))

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
                speA = self.stft(dataA, win, step)
                speB = self.stft(dataB, win, step)
                # speA = np.array(speA, dtype=np.float32).copy()
                # speB = np.array(speB, dtype=np.float32).copy()
                speA /= 32768.0
                speB /= 32768.0
                # img_A = abs(speA[:, : int(fftLen / 2) + 1].T)
                # img_B = abs(speB[:, : int(fftLen / 2) + 1].T)
                img_A = abs(speA.T)
                img_B = abs(speB.T)
                    
                img_A = img_A.tolist()
                img_B = img_B.tolist()    
                
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

class Train():
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 320
        self.channels = 1
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape = (self.img_rows, self.img_cols)

        # Configure data loader
        self.dataset_name = 'materials'
        # Use the DataLoader object to import a preprocessed dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))

        patch = int(self.img_rows / 2**4)
        #(8,8,1)
        self.disc_patch = (patch, 20, 1)

        self.gf = 32
        self.df = 64

        self.lambda_cycle = 10.0
        self.lambda_id = 0.9 * self.lambda_cycle

        # オプティマイザ=最適化アルゴリズム
        optimizer = Adam(0.0002, 0.5)

        # DaとDbを作成し、コンパイル
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # self.d_B.summary()

        # 2つの生成器を作成
        # GabとGbaをインスタンス化
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        # self.g_BA.summary()

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # A→B'→A'' B→A'→B'' サイクル一貫性制約
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
         

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # A→A' and B→B' 同一性損失制約
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # generator のみを学習するために、discriminatorの学習を止める
        self.d_A.trainable = False
        self.d_B.trainable = False

        # generatorが作成した偽物をdiscriminatorに
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
        self.combined.summary()
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=optimizer)


    def build_generator(self):
        d0 = Input(shape=self.img_shape)

        Red0 = Reshape((self.img_rows, self.img_cols, self.channels), input_shape = self.img_shape)(d0)

        d1 = self.conv2d(Red0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)
        d4 = self.conv2d(d3, self.gf * 8)
        u1 = self.deconv2d(d4, d3, self.gf * 4)
        u2 = self.deconv2d(u1, d2, self.gf * 2)
        u3 = self.deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4,strides=1, padding='same', activation='tanh')(u4)
        
        output_img = Reshape((self.img_rows, self.img_cols), input_shape = (512,320,1))(output_img)
        
        model = Model(d0, output_img)
        model.summary()
        return model

    def build_discriminator(self):
        img = Input(shape=self.img_shape)

        Reimg = Reshape((self.img_rows, self.img_cols, self.channels), input_shape = self.img_shape)(img)

        d1 = self.conv2d(Reimg, self.df, normalization=False)
        d2 = self.conv2d(d1, self.df * 2)
        d3 = self.conv2d(d2, self.df * 4)
        d4 = self.conv2d(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        
        model = Model(img, validity)
        model.summary()
        return model

    def sample_images(self, epoch, batch_i):
        r, c = 2, 3
        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./demo/CycleGanDemo/images/%d_%d.png" % (epoch, batch_i))
        # plt.show()
    
    def train(self, epochs, batch_size, sample_interval=50):
        # shape:(batch_size, 8, 8, 1)
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                
                # discriminator
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                fake_A = np.squeeze(fake_A)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                fake_B = np.squeeze(fake_B)
                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # discriminator の合計loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # generator
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

                # if (batch_i % sample_interval == 0):
                    # self.sample_images(epoch, batch_i)
                
                print(self.combined.metrics_names)

                log = ('epoch:{:4}, batch:{:2}, Dloss:{:.4f}, {:.4f}, Gloss:{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}').format(epoch, batch_i, d_loss[0], d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6])
                # print(log)
                # f = open('./demo/CycleGanDemo/log.txt', 'a').write(log+ '\n')
                


class Train(Train):
    @staticmethod
    def conv2d(layer_input, filters, f_size=4, normalization=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                    padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

if __name__ == '__main__':
    t = Train()
    t.train(epochs = 100, batch_size = 8, sample_interval = 10)
# path = os.getcwd()
# print(path)

# path_A = glob.glob('./demo/CycleGanDemo/%s/%sA/*' % ('apple2orange', 'train'))
# path_B = glob.glob('./demo/CycleGanDemo/%s/%sB/*' % ('apple2orange', 'train'))

# print(len(path_A), len(path_B))
 