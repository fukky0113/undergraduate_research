import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import numpy as np
FILENAME="C:\Lab\demo\CycleGanDemovoice\log.txt"

x = np.zeros(1024)
y = np.zeros(1024)
class LogAnalyze():
    # def __init__(self):
        
    def readline_file(self, fd):
        data = fd.readline()
        return data

    def AnalyzeLog(self):
        fd = open(FILENAME)
        dl_ba = []
        ba = []
        cl_ba = []
        ca = []
        rl_ba = []
        nl_ba = []
        tmp = -1

        for _ in range(20):
            data = self.readline_file(fd)
        while(True):
            s1 = ''
            data = self.readline_file(fd)
            if not data:
                break
            
            # if(tmp != data[data.find('epoch') + 9]):
            for i in range(6,12):
                if (data[data.find('Dloss') + i] != ','):
                    s1 += data[data.find('Dloss') + i]
  
            dl_ba.append(float(s1))
            s1 = ''

            for i in range(6,12):
                if(data[data.find('Gloss') + i] != ','):
                    s1 += data[data.find('Gloss') + i]
            ba.append(float(s1))
            s1 = ''

                # for i in range(6,12):
                #     s1 += data[data.find('cl_ba') + i]
                # cl_ba.append(float(s1))
                # s1 = ''

                # for i in range(3,7):
                #     s1 += data[data.find('ca') + i]
                # ca.append(float(s1))                
                # s1 = ''

                # for i in range(6,12):
                #     s1 += data[data.find('rl_ba') + i]
                # rl_ba.append(float(s1))
                # s1 = ''

                # for i in range(6,12):
                #     s1 += data[data.find('nl_ba') + i]
                # nl_ba.append(float(s1))
            
        # ax_1 = plt.subplot(2, 2, 1, xlabel='epoch')
        plt.plot(dl_ba, label = "dl_ba(mse)")
        plt.plot(ba, label = "da(acc)")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
        plt.title('loss1')
        
        # ax_2 = plt.subplot(2, 2, 2, xlabel='epoch')
        # ax_2.plot(cl_ba, label = "cl_ba(mse)")
        # ax_2.plot(ca, label = "ca(acc)")
        # ax_2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
        # plt.title('loss2')

        # ax_3 = plt.subplot(2, 2, 3, xlabel='epoch')
        # ax_3.plot(rl_ba, label = "rl_ba(mae)")
        # ax_3.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
        # plt.title('loss3')

        # ax_4 = plt.subplot(2, 2, 4, xlabel='epoch')
        # ax_4.plot(nl_ba, label = "nl_ba(mae)")
        # ax_4.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=10)
        # plt.title('loss4')


        plt.tight_layout()  # タイトルの被りを防ぐ
        plt.show()

if __name__ == "__main__":
    print("=== readlines() test ===")
    a = LogAnalyze()
    a.AnalyzeLog()