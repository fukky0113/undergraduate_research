import numpy as np
from scipy import ceil, complex64, float64, hanning, zeros
from scipy.io import wavfile

import matplotlib.pyplot as plt

import glob

#短時間フーリエ変換(データ,長さ?フレーム?用語の名前が分からん)
def stft(x, win, step):
    l = len(x)
    N = len(win)
    M = int(ceil(float(l - N + step)/ step))

    new_x = zeros(N + ((M - 1) * step), dtype = float64)
    # new_x = [0] * (N + ((M - 1) * step))
    new_x[: l] = x 
    X = zeros([M, N], dtype = complex64)
    for m in range(M):
        start = step * m
        X[m, :] = np.fft.fft(new_x[start : start + N] * win)
    return X



def load_batch(batch_size=1, is_testing=False):
    fftLen = 1024 / 2
    win = np.hamming(fftLen)
    step = fftLen / 4

    dataset_name = 'materials'

    data_type = "data_"
    path_A = glob.glob('./demo/CycleGanDemovoice/%s/data/New_%sA/*.wav' % (dataset_name, data_type))
    path_B = glob.glob('./demo/CycleGanDemovoice/%s/data/New_%sB/*.wav' % (dataset_name, data_type))

    n_batches = int(min(len(path_A), len(path_B)) / batch_size)
    total_samples = n_batches * batch_size

    path_A = np.random.choice(path_A, total_samples, replace=False)
    path_B = np.random.choice(path_B, total_samples, replace=False)

    for i in range(n_batches - 1):    
        batch_A = path_A[i*batch_size:(i+1)*batch_size]
        batch_B = path_B[i*batch_size:(i+1)*batch_size]
        imgs_A = []
        imgs_B = []

        for img_A, img_B in zip(batch_A, batch_B):
            rateA, dataA = wavfile.read(img_A)
            rateB, dataB = wavfile.read(img_B)
            dataA = np.ravel(dataA)
            dataB = np.ravel(dataB)
            speA = stft(dataA, win, step)
            speB = stft(dataB, win, step)
            speA = np.array(speA, dtype=np.float32).copy()
            speB = np.array(speB, dtype=np.float32).copy()
            speA /= 32768.0
            speB /= 32768.0
            # img_A = abs(speA[:, : int(fftLen / 2) + 1].T)
            # img_B = abs(speB[:, : int(fftLen / 2) + 1].T)
            img_A = abs(speA.T)
            img_B = abs(speB.T)   
            a = img_A[:, :, np.newaxis]
            b = img_B[:, :, np.newaxis]     
            # print(a.shape)
            imgs_A.append(a)
            imgs_B.append(b)

        yield imgs_A, imgs_B


def load_data(domain, batch_size=1, is_testing=False):
    dataset_name = 'materials'
    path = glob.glob('./demo/CycleGanDemovoice/%s/data/data_%s/*.wav' % (dataset_name, domain))
    batch_images = np.random.choice(path, size=batch_size)

    imgs = []
    for img_path in batch_images:
        data = wavfile.read(img_path)
        # if not is_testing:
        #     img = scipy.misc.imresize(img, self.img_res)
        #     if np.random.random() > 0.5:
        #         img = np.fliplr(img)
        # else:
        # img = scipy.misc.imresize(img, self.img_res)
        data = np.ravel(data)
        spe = stft(data, win, step)
        spe = np.array(spe, dtype=np.float32).copy()
        spe /= 32768.0
        img = abs(spe[:, : int(fftLen / 2) + 1].T)
        imgs.append(img)
    return imgs

def istft(X, win, step):
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."
    l = (M - 1) * step + N
    x = zeros(l, dtype = float64)
    wsum = zeros(l, dtype = float64)
    for m in range(M):
        start = step * m
        x[start : start + N] = x[start : start + N] + np.fft.ifft(X[m, :]).real * win
        wsum[start : start + N] += win ** 2
    pos = (wsum != 0)
    x_pre = x.copy()
    x[pos] /= wsum[pos]
    return x


def cut5120(data):
    datalen = len(data)
    random = np.random.randint(0, datalen - 5120)
    print(datalen/16000)
    # return data[random : random + 5120]
    return data[0 : 2560]

def GriffinLim(a, win, step, iterations=100):
    approximated_signal=None
    for k in range(iterations):
        if approximated_signal is None:
            _P = np.random.randn(*a.shape)
        else:
            _D = stft(approximated_signal, win, step)
            _P = np.angle(_D)
        _D = a * np.exp(1j * _P)
        approximated_signal = istft(_D, win, step)
    return approximated_signal


# LOAD_FILE = r'C:\Lab\CycleGAN_LAB\data\data_B\b.wav'
# LOAD_FILE = r"C:\Lab\demo\CycleGanDemovoice\materials\data\NormalizationA\あなたが嬉しいと、わたしも嬉しいです.wav"
# LOAD_FILE = r"C:\Lab\demo\CycleGanDemovoice\materials\data\New_data_A\あなたが嬉しいと、わたしも嬉しいです.wav"
LOAD_FILE = './demo/CycleGanDemovoice/materials/data/demo.wav'
fftLen = 128 - 2
win = np.hanning(fftLen)
step = fftLen // 4
rate, data = wavfile.read(LOAD_FILE)
data = np.ravel(data)

# print(data)
# data = np.pad(data, [1280,], 'constant')
# print(data)

path = glob.glob('./demo/CycleGanDemoVoiceV2/materials/data/NB2/*.wav')

for i in path:
    
    rate, data = wavfile.read(i)
    # print(i[46:])
    wavfile.write('./demo/CycleGanDemoVoiceV2/materials/data/tmp/' + i[46:], 16000, (data).astype(np.int16))

# data = np.array(data, dtype=np.float64).copy()

# data = cut5120(data)
# # data /= 32768.0
# spe = stft(data, win, step)
# spe = np.array(spe, dtype=np.float64).copy()




# halfspe = abs(spe[:, : fftLen // 2 + 1].T)
# # halfspe = abs(spe.T)

# print(halfspe.shape)
# print(halfspe)

# plt.imshow(halfspe, cmap='gray', origin = "lower", aspect="auto")
# plt.show()


# # SET = 2500

# # print('mae:', halfspe)
# # halfspe = np.clip(halfspe, SET, None)

# # halfspe = (np.log(halfspe) - np.log(SET)) / (np.log(4161536) - np.log(SET)) 



# halfspe = np.exp(halfspe * (np.log(4161536) - np.log(SET)) + np.log(SET))

# # print('ato:', halfspe)
# halfspe[halfspe < SET] = 1

# print('ato:', halfspe)


# print(halfspe,np.max(halfspe))



# print(halfspe.shape)
# print(len(data))

# print(rate,spe.shape)

# print(halfspe)

# print(halfspe.shape)

# print(backhalfspe)

# print(restorespe.T)

# a = GriffinLim(halfspe.T, win, step, iterations=100)

# print('a:',a)
# a = np.exp(a * (np.log(4161536) - np.log(SET)) + np.log(SET))

#     # src[src < 1000] = 1
# print('a:',a)

# plt.plot(a)
# plt.ylim(-10,10)
# plt.show()
# wavfile.write('./demo/CycleGanDemovoice/demoresult/demo.wav' , 16000, (a).astype(np.int16))
# wavfile.write('./demo/CycleGanDemovoice/demoresult/demo.wav', rate, (data).astype(np.int16))




# halfspe = (np.log(8000)) / 9

# print(np.max(spe))

# spe = np.exp(halfspe * 9)

# print(spe)