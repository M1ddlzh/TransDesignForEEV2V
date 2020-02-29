"""
This is the environment for tow-users vehicles network.
"""

import numpy as np

np.random.seed(0)

class DQNEnvTwoUsers:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        
        # 两对车间的距离、信道参数
        self.dii = 10
        self.d21 = 100
        self.d12 = 120
        self.N0 = 10**-15   # W/Hz
        self.band_width = 1e5    # Hz, 100kHz
        self.PLii = -(103.4+24.2*np.log10(self.dii/1000))    # Km, dBm
        self.PL12 = -(103.4+24.2*np.log10(self.d21/1000))
        self.PL21 = -(103.4+24.2*np.log10(self.d12/1000))
        self.sigmaii = (10**(self.PLii/10))      # 经过大尺度衰落后功率缩小多少倍，期望
        self.sigma12 = (10**(self.PL12/10))
        self.sigma21 = (10**(self.PL21/10))
        
        self.hii_intervals_len = 1.8e-5 / 100   # max / num intervals
        self.h12_intervals_len = 7.6e-8 / 100
        self.h21_intervals_len = 4.7e-8 / 100

        # 信息产生参数
        self.ty1_gene_rate = 200      # type1以固定速率产生, kbps, 
        self.ty2_lamda = 500              # type2产生速率的均值, kbps 

        # 传输限制参数
        self.rate_min = 0.      # kbps
        self.rate_max = 1000.   
        self.power_min = 0. + 1e-10  # TODO:由于是dBm形式，又需要加减操作，因此最小设为0，设为负值需要改求能效比的代码
        self.power_max = 23.
        
        # 奖励函数参数
        self.phi_0 = 0.9            # 需要和下面函数的参数self.n_xx一起设计
        self.phi_max = 0.95
        self.n_ts1 = 20              # 一次函数的系数
        self.n_ts2 = -1. - self.n_ts1 * self.phi_0           # 过(phi_0, -1)
        self.n_ts3 = -2
        self.n_ts4 = -0.1 - self.n_ts3                      # 过(1, -0.1)

        self.Q_min = 5 * self.ty2_lamda
        self.Q_max = 10 * self.ty2_lamda
        self.Q_intervals_len = 30
        self.n_q2 = -0.1                                    # 过(0, -0.1)
        self.n_q1 = -self.n_q2 / self.Q_min
        self.n_q3 = -1
        self.n_q4 = -1 - self.n_q3 * self.Q_max             # 过(Q_max, -1)

        self.alpha = 0.01            # 能效
        self.beta = 10             # type1成功率, 10/-8/8
        self.gamma = 0.003            # Q长度

    def get_channel_h(self):
        h11 = np.random.gamma(2, self.sigmaii / 2)    # if h ~ Nakagami-m, h^2 ~ gamma; here h11 is h^2
        h22 = np.random.gamma(2, self.sigmaii / 2)
        h12 = np.random.gamma(2, self.sigma12 / 2)
        h21 = np.random.gamma(2, self.sigma21 / 2)
        # h11, h22max: 1.8e-5; h12max: 7.6e-8; h21max: 4.7e-8
        h11 = h11 // self.hii_intervals_len * self.hii_intervals_len + self.hii_intervals_len / 2   # 离散化
        h22 = h22 // self.hii_intervals_len * self.hii_intervals_len + self.hii_intervals_len / 2
        h12 = h12 // self.h12_intervals_len * self.h12_intervals_len + self.h12_intervals_len / 2
        h21 = h21 // self.h21_intervals_len * self.h21_intervals_len + self.h21_intervals_len / 2
        H = [h11, h22, h12, h21]
        return H

    def get_reward(self, trans_sucess, r, p, I, ty1_succ_rate, Q_len, energy_effi):
        """
        计算奖励
        """
        # TODO: 要保证两对用户都满足传输限制时的奖励要高于只有一对用户满足的情况，当然前提是两对用户确实可以都满足
        # energy efficiency reward 
        # r_ee = trans_sucess * r - energy_effi * p   
        r_ee = energy_effi    
        # TODO: 直接试试energy_effi作为reward；p是dBm单位，小于0的话会出错

        # type1 transfer rate reward
        # if ty1_succ_rate < self.phi_0:
        #     r_ts = self.n_ts1 * ty1_succ_rate + self.n_ts2
        # elif self.phi_0 <= ty1_succ_rate < self.phi_max:
        #     r_ts = 10.
        # else:
        #     r_ts = self.n_ts3 * ty1_succ_rate + self.n_ts4

        if ty1_succ_rate < self.phi_0:
            r_ts = ty1_succ_rate
        elif self.phi_0 <= ty1_succ_rate < self.phi_max:
            r_ts = self.phi_0
        else:
            r_ts = self.n_ts3 * ty1_succ_rate + self.n_ts4


        # type2 queue length reward
        if Q_len < self.Q_min:
            r_q = self.n_q1 * Q_len + self.n_q2
        elif self.Q_min <= Q_len < self.Q_max:
            r_q = self.Q_min
        else:
            r_q = self.n_q3 * Q_len + self.n_q4
        
        # final reward
        reward = self.alpha * r_ee + self.beta * r_ts + self.gamma * r_q

        return reward

    def get_feedback(self, r, p, I, ty2_gene_rate, ty1_sum_succ, ty1_sum_fail, Q_len, sum_r, sum_p):
        """
        根据发射速率和信道容量的大小关系得到奖励，并更新环境状态
        """
        if r > I:                           # 发送数据超出信道容量，传输失败
            trans_sucess = 0
            ty1_sum_fail += 1
            ty1_succ_rate = ty1_sum_succ / (ty1_sum_succ + ty1_sum_fail)

            Q_len += ty2_gene_rate

            sum_p += p
            energy_effi = sum_r / sum_p
            reward = self.get_reward(trans_sucess, r, p, I, ty1_succ_rate, Q_len, energy_effi)
        elif self.ty1_gene_rate <= r <= I:               # 可以同时发type1和type2
            trans_sucess = 1
            ty1_sum_succ += 1
            ty1_succ_rate = ty1_sum_succ / (ty1_sum_succ + ty1_sum_fail)

            Q_len = max(0., Q_len + ty2_gene_rate - (r - self.ty1_gene_rate))

            sum_r += r; sum_p += p
            energy_effi = sum_r / sum_p
            reward = self.get_reward(trans_sucess, r, p, I, ty1_succ_rate, Q_len, energy_effi)
        else:                               # 只发送type2
            trans_sucess = 1
            ty1_sum_fail += 1
            ty1_succ_rate = ty1_sum_succ / (ty1_sum_succ + ty1_sum_fail)

            Q_len = max(0., Q_len + ty2_gene_rate - r)

            sum_r += r; sum_p += p
            energy_effi = sum_r / sum_p
            reward = self.get_reward(trans_sucess, r, p, I, ty1_succ_rate, Q_len, energy_effi)

        return reward, ty1_sum_succ, ty1_sum_fail, Q_len, sum_r, sum_p, energy_effi, trans_sucess

    def step(self, info):
        """
        执行动作，返回下一状态
        """
        a, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
            Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2 = info

        p1, p2, r1, r2 = a
        p1 = (p1 - self.n_actions/8) / (self.n_actions/8)       # index -> [-1, 1]
        p2 = (p2 - self.n_actions/4 - self.n_actions/8) / (self.n_actions/8)
        r1 = (r1 - self.n_actions/2 - self.n_actions/8) / (self.n_actions/8)
        r2 = (r2 - (self.n_actions/4)*3 - self.n_actions/8) / (self.n_actions/8)

        p1 = p1 * (self.power_max / 2) + (self.power_max / 2) + 1e-15       # [-1, 1] -> dBm
        p2 = p2 * (self.power_max / 2) + (self.power_max / 2) + 1e-15
        r1 = r1 * (self.rate_max / 2) + (self.rate_max / 2) + 1e-15
        r2 = r2 * (self.rate_max / 2) + (self.rate_max / 2) + 1e-15

        h11, h22, h12, h21 = H
        p1_w = 10 ** (p1 / 10) / 1000        # dBm -> W
        p2_w = 10 ** (p2 / 10) / 1000
        I1 = self.band_width * np.log2(1 + h11 * p1_w / (self.band_width * self.N0 + h12 * p2_w)) / 1e3     # bit -> kbit
        I2 = self.band_width * np.log2(1 + h22 * p2_w / (self.band_width * self.N0 + h21 * p1_w)) / 1e3     # bit -> kbit

        # type2产生速率
        ty2_gene_rate_1 = np.random.poisson(self.ty2_lamda)
        ty2_gene_rate_2 = np.random.poisson(self.ty2_lamda)

        reward_1, ty1_sum_succ_1, ty1_sum_fail_1, Q_len_1, sum_r_1, sum_p_1, \
        energy_effi_1, trans_sucess_1 = \
            self.get_feedback(r1, p1, I1, ty2_gene_rate_1, ty1_sum_succ_1, ty1_sum_fail_1, 
                               Q_len_1, sum_r_1, sum_p_1)

        reward_2, ty1_sum_succ_2, ty1_sum_fail_2, Q_len_2, sum_r_2, sum_p_2, \
        energy_effi_2, trans_sucess_2 = \
            self.get_feedback(r2, p2, I2, ty2_gene_rate_2, ty1_sum_succ_2, ty1_sum_fail_2, 
                               Q_len_2, sum_r_2, sum_p_2)

        Q_len_1 = Q_len_1 // self.Q_intervals_len * self.Q_intervals_len + self.Q_intervals_len / 2      # 离散化
        Q_len_2 = Q_len_2 // self.Q_intervals_len * self.Q_intervals_len + self.Q_intervals_len / 2

        next_state = np.array([ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1),
                               ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2), 
                               Q_len_1, Q_len_2, 
                               h11, h22, h12, h21, trans_sucess_1, trans_sucess_2])   
                               # TODO: 增加额外信息如type1、2产生速率，发射功率

        reward = (reward_1 + reward_2) / 2

        done = False

        H = self.get_channel_h()

        return next_state, reward, done, H, \
               ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
               Q_len_1, Q_len_2, \
               sum_r_1, sum_p_1, sum_r_2, sum_p_2, energy_effi_1, energy_effi_2

    def reset(self):
        """
        获得初始状态，并初始化第一次step()的参数
        """
        initial_succ_rate_1 = np.random.uniform(low=self.phi_0, high=self.phi_max)
        initial_succ_rate_2 = np.random.uniform(low=self.phi_0, high=self.phi_max)
        ty1_sum_succ_1 = int(initial_succ_rate_1 * 100)
        ty1_sum_fail_1 = 100 - ty1_sum_succ_1
        ty1_sum_succ_2 = int(initial_succ_rate_2 * 100)
        ty1_sum_fail_2 = 100 - ty1_sum_succ_2
        Q_len_1 = np.random.randint(low=self.Q_min, high=self.Q_max) \
            // self.Q_intervals_len * self.Q_intervals_len + self.Q_intervals_len / 2      # 离散化
        Q_len_2 = np.random.randint(low=self.Q_min, high=self.Q_max)\
            // self.Q_intervals_len * self.Q_intervals_len + self.Q_intervals_len / 2
        H = self.get_channel_h()
        h11, h22, h12, h21 = H
        trans_sucess_1 = 0   # 防止初始时选一个很大的r，很小的p，造成能效很大
        trans_sucess_2 = 0
        
        state = np.array([ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1),   # 0%~100%，已经离散
                          ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2),
                          Q_len_1, Q_len_2, 
                          h11, h22, h12, h21, trans_sucess_1, trans_sucess_2])
        
        sum_r_1 = np.random.uniform(low=self.rate_min, high=self.rate_max)
        sum_p_1 = np.random.uniform(low=self. power_min, high=self.power_max)
        sum_r_2 = np.random.uniform(low=self.rate_min, high=self.rate_max)
        sum_p_2 = np.random.uniform(low=self.power_min, high=self.power_max)

        return state, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
               Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2 
