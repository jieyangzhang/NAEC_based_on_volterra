"""
基于论文《Nonlinear Acoustic Echo Cancellation Based on Volterra Filters》进行仿真
date : 2021年 10月 23日 星期六 15:16:36 CST
"""
 
from cProfile import label
import numpy as np
from numba import jit
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time
 
class NAEC_BASED_ON_VOLITERRA:
    """
    func : 初始化
    param x : 远端参考信号的时域序列
    param d : 近端采集信号的时域序列
    """
    def __init__(self, x, d):
        self.N = 128                                                # 线性滤波长度，参考论文IV小节:128
        self.L = 10                                                 # Volterra滤波非线性记忆长度，参考论文IV小节:10
        self.L2 = int(self.L * (self.L + 1) / 2)
        self.L3 = int(self.L * (self.L + 1) * (self.L + 2) / 6)
        self.sample_rate = 8000                                     # 默认使用8k采样率
        self.u1 = 0.5                                               # 线性滤波更新步长，参考论文IV小节
        # Volterra二阶核的更新步长，参考论文IV小节
        self.u2 = 0.05
        # Volterra三阶核的更新步长，参考论文IV小节
        self.u3 = 0.05
        self.sigma_w = 0                                            # 计算方式参考论文公式（46）
        # 线性滤波参数稳定状态， 1--稳定， 0--不稳定
        self.w_steady_state = 0
        # 存在非线性失真， 1--存在， 0--不存在
        self.nonlinear_exist = 0
        L = self.L
        N = self.N
        # 线性滤波器参数，shape = [N, 1]
        self.w = np.zeros([N, 1], dtype=np.float)
        # 线性滤波器参数的均值，shape = [N, 1]
        self.bar_w = np.zeros([N, 1], dtype=np.float)
        # 二阶Volterra非线性滤波器参数，shape = [L * (L + 1) / 2, 1]
        self.h2 = np.zeros([self.L2, 1], dtype=np.float)
        # 三阶Volterra非线性滤波器参数，shape = [L * (L + 1) * (L + 2) / 6, 1]
        self.h3 = np.zeros([self.L3, 1], dtype=np.float)
        self.x = x
        self.d = d
        self.pos = N + L - 2
        self.data_len = min(len(x), len(d))
        self.r = np.zeros(self.data_len, dtype=np.float)
        self.e = np.zeros(self.data_len, dtype=np.float)
        self.d_hat = np.zeros(self.data_len, dtype=np.float)
        self.U2 = np.zeros([N, self.L2], dtype=np.float)            # 参考式（10）
        self.U3 = np.zeros([N, self.L3], dtype=np.float)
        self.x_nl = np.zeros([N, 1], dtype=np.float)
 
        return
 
    def __w_steady_state_detector(self):
        """
        func :  判断线性滤波器的更新状态， 基于sigma_w和阈值的对比；当sigma_w小于阈值，判定为稳定状态，允许更新
                非线性核；当sigma_w大于阈值，判定为非稳定，不允许更新非线性核；sigma_w的计算方式参考论文公式（46）
        """
        # fixme，论文未给出alpha, belta_up, belta_down的详细值，close to 1 & belta_down <= belta_up
        alpha = 0.95
        belta_up = 0.99
        belta_down = 0.95
        sigma_w_threshold = 1.1                                     # 判定线性滤波参数稳定的阈值，参考论文IV节
        self.bar_w = alpha * self.bar_w + (1 - alpha) * self.w      # Eq.45
        # w的均值和瞬时值差值的L1范数, 论文描述使用l1范数和l2范数均可
        # Eq.46
        norm_l1 = np.linalg.norm((self.bar_w - self.w), ord=1)
        if norm_l1 >= self.sigma_w:
            self.sigma_w = belta_up * self.sigma_w + (1 - belta_up) * norm_l1
        else:
            self.sigma_w = belta_down * self.sigma_w + (1 - belta_down) * norm_l1
        self.w_steady_state = 1 if self.sigma_w < sigma_w_threshold else 0
        # print('sigma_w ：', self.sigma_w)
        return
 
    def __nonlinear_state_detector(self):
        threshold = 0   # TODO, 确认非线性失真检测阈值
        element = np.linalg.norm(np.dot(self.U2.T, self.w), ord=2) ** 2
        denominator = np.linalg.norm(self.w, ord=2) ** 2
        self.r[self.pos] = element / denominator   # Eq.47
        # print('r=%f ele=%f deno=%f' % (r, element, denominator))
        self.nonlinear_exist = 1 if (self.r[self.pos] > threshold) else 0
 
        return
 
    def __get_U2(self, x):
        """
        func : 计算U2, 参考论文公式（10）
        param x : 远端信号序列，长度为self.L + self.N - 1
        """
        if self.pos == self.N + self.L - 2:
            for i in range(self.N):
                row_idx = 0
                for j in range(self.L):
                    for k in range(self.L - j):
                        self.U2[i][row_idx] = x[-i-1-j] * x[-i-1-j-k]
                        row_idx += 1
        else:
            self.U2[1:self.N] = self.U2[0:self.N-1]
            row_idx = 0
            for j in range(self.L):
                for k in range(self.L - j):
                    self.U2[0][row_idx] = x[-1-j] * x[-1-j-k]
                    row_idx += 1
        return
 
    def __get_U3(self, x):
        """
        func : 计算U3
        param x : 远端信号序列，长度为self.L + self.N - 1
        """
        if self.pos == self.N + self.L - 2:
            for i in range(self.N):
                row_index = 0
                for j in range(self.L):
                    for k in range(self.L - j):
                        for l in range(self.L - j - k):
                            self.U3[i][row_index] = x[-i-1-j] * x[-i-1-j-k] * x[-i-1-j-k-l]
                            row_index += 1
        else:
            self.U3[1:self.N] = self.U3[0:self.N-1]
            row_index = 0
            for j in range(self.L):
                for k in range(self.L - j):
                    for l in range(self.L - j - k):
                        self.U3[0][row_index] = x[-1-j] * x[-1-j-k] * x[-1-j-k-l]
                        row_index += 1
        return
 
    def __update_w(self):
        """
        func : 更新Volterra线性滤波器w参数
        """
        norm2 = np.linalg.norm(self.x_nl, ord=2)                                  # 非线性向量二范数
        self.w = self.w + self.u1 * self.x_nl * self.e[self.pos] / (norm2 ** 2)   # Eq.24
 
        return
 
    def __update_h2(self):
        """
        func : 更新Volterra二阶非线性核
        """
        element = np.dot(self.U2.T, self.w)
        norm2 = np.linalg.norm(element, ord=2)
        self.h2 = self.h2 + self.u2 * self.e[self.pos] * element / (norm2 ** 2) # Eq.32
 
        return
 
    def __update_h3(self):
        """
        func : 更新Volterra三阶非线性核
        """
        element = np.dot(self.U3.T, self.w)
        norm2 = np.linalg.norm(element, ord=2)
        self.h3 = self.h3 + self.u3 * self.e[self.pos] * element / (norm2 ** 2) # Eq.33
 
        return
 
    def __forward(self):
        """
        func : 计算预估的非线性回声，并从近端信号取出得到残留回声
        param x :
        """
        x_tmp = self.x[self.pos - self.N - self.L + 2 : (self.pos + 1)].copy()
        # 计算非线性部分
        self.__get_U2(x_tmp)
        self.__get_U3(x_tmp)
        x_1 = x_tmp[-self.N::].reshape((self.N, 1))
        x_2 = np.dot(self.U2, self.h2)                              # Eq.11
        x_3 = np.dot(self.U3, self.h3)
        self.x_nl = x_1 + x_2+ x_3                                  # Eq.3
        # 计算线性部分,得到预估的回声
        self.d_hat[self.pos] = np.dot(self.x_nl.T, self.w)          # Eq.6
        # 计算误差信号
        self.e[self.pos] = self.d[self.pos] - self.d_hat[self.pos]  # Eq.7
 
        return
 
    def __update(self):
        """
        func : 更新Volterra滤波器
        """
        '''
        更新非线性核的策略：
        1. 远端信号功率足够大以产生非线性失真
        2. 线性滤波器更性状态平稳
 
        为了追踪回声路径的变化，线性滤波器始终处于更新状态
        '''
        self.__w_steady_state_detector()
        self.__nonlinear_state_detector()
        if self.nonlinear_exist and self.w_steady_state :
            self.__update_h2()
            self.__update_h3()
        self.__update_w()
 
        return
 
    def process(self):
        avg_time = 0
        while self.pos < self.data_len:
            time_start = time.time()
            self.__forward()
            self.__update()
            avg_time += time.time() - time_start
            self.pos += 1
            print('processing:%f %%, time per sample:%f s' % \
                        (self.pos / self.data_len * 100, avg_time / (self.pos - self.N + self.L - 2)), \
                        end='\r', flush=True)
            if self.pos % 80000 == 0:
                plt.subplot(211)
                plt.plot(self.d[0 : self.pos], label='d')
                plt.plot(self.d_hat[0 : self.pos], label='d_hat')
                plt.legend()
                plt.subplot(212)
                plt.plot(self.e[0 : self.pos], label='e')
                plt.legend()
                # plt.subplot(313)
                # plt.plot(self.r[0 : self.pos], Label='r')
                # plt.legend()
                plt.show()
        # self.e *= (2 ** (15) - 1)
        # self.d_hat *= (2 ** (15) - 1)
        wavfile.write("./e.wav", 8000, self.e.astype(np.int16))
        wavfile.write("./d_hat.wav", 8000, self.d_hat.astype(np.int16))
 
        return
 
if __name__ == "__main__":
    _, d = wavfile.read("./near_part.wav")
    _, x = wavfile.read("./far_part.wav")
    max_value = (2 ** (15) - 1)
    # d = d / max_value
    # x = x / max_value
    aec = NAEC_BASED_ON_VOLITERRA(x, d)
    aec.process()