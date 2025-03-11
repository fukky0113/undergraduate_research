import numpy as np
from scipy import ceil, complex64, float64, hanning, zeros
from scipy.io import wavfile
import matplotlib.pyplot as plt
import glob

MIN = 41250

path_A = glob.glob('C:\Lab\demo\CycleGanDemovoice\materials\data\data_A\*')
path_B = glob.glob('C:\Lab\demo\CycleGanDemovoice\materials\data\data_B\*')

imgs_A, imgs_B = [], []

for img_A, img_B in zip(path_A, path_B):
    nameA = ''
    devsublenA = 0
    nameB = ''
    devsublenB = 0
    rateA, dataA = wavfile.read(img_A)
    rateB, dataB = wavfile.read(img_B)

    lenA = len(dataA) / MIN
    ceillenA = int(np.ceil(lenA))
    widthceillenA = MIN * ceillenA

    sublenA = widthceillenA - len(dataA)
    devsublenA = sublenA // (ceillenA-1)
    # print(devsublenA, sublenA // ceillenA)

    for i in range(50):
        # print(img_A[52 + i :53 + i])
        if (img_A[52 + i :53 + i] == '.'):
            nameA = img_A[52:52+i]
            break

    lenB = len(dataB) / MIN
    ceillenB = int(np.ceil(lenB))
    widthceillenB = MIN * ceillenB

    sublenB = widthceillenB - len(dataB)
    devsublenB = sublenB // (ceillenB-1)

    for i in range(50):
        if (img_B[52 + i :53 + i] == '.'):
            nameB = img_B[52:52+i-1]
            break

    for i in range(ceillenA):
        a = dataA[i*(MIN-devsublenA):i*(MIN-devsublenA)+MIN]

        # print(nameA ,len(dataA), devsublenA, i*(MIN-devsublenA),i*(MIN-devsublenA)+MIN)
        wavfile.write(r"C:\Lab\demo\CycleGanDemovoice\materials\data\New_data_A\\" + nameA + str(i) + ".wav", rateA, a)

    for i in range(ceillenB):
        a = dataB[i*(MIN-devsublenB):i*(MIN-devsublenB)+MIN]

        # print(len(dataB), devsublenB, i*(MIN-devsublenB),i*(MIN-devsublenB)+MIN)
        wavfile.write(r"C:\Lab\demo\CycleGanDemovoice\materials\data\New_data_B\\" + nameB + str(i) + ".wav", rateB, a)