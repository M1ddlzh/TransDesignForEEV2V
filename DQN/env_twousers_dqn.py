import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class EnvTwoUsers(gym.Env):
    """
    Two Users Env
    """
    def __init__(self, steps_in_one_episode):
        self.step_in_one_episode = steps_in_one_episode
        self.step_now = 0
        ############### vehicles distance parameters ###############
        # # self.dii = 20
        # # self.d21 = 50
        # # self.d12 = 90
        # self.N0 = 0.04               # Watt, total noise in bandwidth
        # self.band_width = 1e5           # Hz
        # # self.PLii = -(103.4+24.2*np.log10(self.dii/1000))    # m -> km, Path Loss: dB
        # # self.PL12 = -(103.4+24.2*np.log10(self.d21/1000))
        # # self.PL21 = -(103.4+24.2*np.log10(self.d12/1000))
        # self.sigmaii = 0.5  # (10**(self.PLii/10))      
        # self.sigma12 = 0.5  # (10**(self.PL12/10))
        # self.sigma21 = 0.5  # (10**(self.PL21/10))
        # self.m = 2

        # self.dii = 10
        # self.d12 = 10
        # self.d21 = self.d12 + 2 * self.dii
        # self.PLii = -(33 + 22.7 * np.log10(self.dii))    # m, Path Loss: dB
        # self.PL12 = -(33 + 22.7 * np.log10(self.d12))
        # self.PL21 = -(33 + 22.7 * np.log10(self.d21))
        # self.sigmaii = (10 ** (self.PLii / 10))
        # self.sigma12 = (10 ** (self.PL12 / 10))
        # self.sigma21 = (10 ** (self.PL21 / 10))
        self.N0 = 1           
        # sigmaii = 0.5
        # sigma12 = 0.5
        # sigma21 = 0.5
        # N0 = 0.04
        self.band_width = 1                        # 100 kHz
        # self.m = 1
        self.scale_ii = np.sqrt(1 / 2)
        self.scale_ij = np.sqrt(0.03 / 2)

        ############### amount of date parameters ###############
        self.ty1_gene_rate = 0.5        # type1 generate rate, kbps
        self.ty2_lamda = 1           # type2 poission mean, kbps 

        ############### reward function parameters ###############
        self.phi_0 = 1            
        self.Q_max = 10 * self.ty2_lamda
        self.alpha = 100                  # ee reward weight 
        self.beta = 10                   # ty1 reward weight 
        self.gamma = 2                  # queue length weight 

        ############### action and state parameters ###############
        self.steps_for_plot = 40000
        self.step_until_now = 0
        self.all_actions = []
        # self.rate_min = 0.              # kbps
        # self.rate_max = 1000.   
        self.power_min = 0
        self.power_max = 50
        # self.action_min = np.array([self.power_min, self.power_min, self.rate_min, self.rate_min])
        # self.action_max = np.array([self.power_max, self.power_max, self.rate_max, self.rate_max])
        # self.range = 10
        # self.a_min = -self.range
        # self.a_max = self.range
        # self.action_min = np.array([self.a_min]*2, dtype=np.float32)
        # self.action_max = np.array([self.a_max]*2, dtype=np.float32)
        # # self.action_space_conti = spaces.Box(
        # self.action_space = spaces.Box(
        #     low=self.action_min, high=self.action_max)
        # # self.action_space_discre = spaces.Discrete(4)   # two users, each can transmit or not transmit
        # self.observation_space = spaces.Box(
        #     low=0., high=float("inf"), shape=(7,), dtype=np.float32)
        self.action = list(range(0, 51, 2)) 
        self.n_actions = len(self.action)
        self.n_features = 6

        ############### simple state features scaling parameters ###########
        self.s_amp = 10
        self.Q_shr = 10 * self.ty2_lamda
        self.h_amp_self = 1
        self.h_amp_betw = 100

        ############### counter of rate and channle capacity ###############
        self.situation_1 = 0
        self.situation_2 = 0
        self.situation_3 = 0

        self.large_than_Q_1 = 0
        self.large_than_Q_2 = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_channel_h(self):
        """
        get channel parameters
        """
        # h11 = self.np_random.gamma(self.m, self.sigmaii / self.m)    
        # h22 = self.np_random.gamma(self.m, self.sigmaii / self.m)
        # h12 = self.np_random.gamma(self.m, self.sigma12 / self.m)
        # h21 = self.np_random.gamma(self.m, self.sigma21 / self.m)
        h11 = (self.np_random.rayleigh(self.scale_ii)) ** 2
        h22 = (self.np_random.rayleigh(self.scale_ii)) ** 2
        h12 = (self.np_random.rayleigh(self.scale_ij)) ** 2
        h21 = (self.np_random.rayleigh(self.scale_ij)) ** 2

        H = [h11, h22, h12, h21]
        return H

    def get_reward(self, is_ty1_trans, ty1_succ_rate, Q_len, 
        energy_effi_previous, energy_effi, p):
        """
        get reward
        """
        # TODO: 要保证两对用户都满足传输限制时的奖励要高于只有一对用户满足的情况，当然前提是两对用户确实可以都满足

        # energy efficiency reward 
        r_ee = energy_effi - energy_effi_previous # - p

        # type1 transfer rate reward
        # if self.step_now == self.step_in_one_episode \
        #     and ty1_succ_rate < self.phi_0:
        #     r_ts = -100
        # elif ty1_succ_rate < self.phi_0:
        #     r_ts = 7 * is_ty1_trans
        # else:
        #     r_ts = 0
        r_ts = -10 + 10 * is_ty1_trans

        # type2 queue length reward
        if Q_len <= self.Q_max:
            r_ql = 0
        else:
            r_ql = -10

        # final reward
        reward = self.alpha * r_ee + self.beta * r_ts + self.gamma * r_ql

        return reward

    def get_feedback(self, p, trans_ty1, I, ty2_gene_rate, ty1_sum_succ, 
        Q_len, sum_r, sum_p):
        """
        two cases of transmission
        """
        if self.ty1_gene_rate <= I:

            # if trans_ty1:           # transfer ty1 and ty2 both
            #     self.situation_1 += 1
            #     is_ty1_trans = 1
            #     ty1_sum_succ += 1
            #     ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
            #     Q_len = max(
            #         0., Q_len + ty2_gene_rate - (I - self.ty1_gene_rate))
            # else:                   # transfer ty2 only
            #     self.situation_2 += 1
            #     is_ty1_trans = 0
            #     ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
            #     Q_len = max(0., Q_len + ty2_gene_rate - I)
            energy_effi_previous = sum_r / sum_p
            sum_r += min(I, self.ty1_gene_rate + Q_len)
            sum_p += p
            self.situation_1 += 1
            is_ty1_trans = 1
            ty1_sum_succ += 1
            ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
            Q_len = max(
                0., Q_len - (I - self.ty1_gene_rate)) + ty2_gene_rate

            # if ty1_sum_succ / self.step_in_one_episode < self.phi_0 * 1.05:    # transfer ty1 and ty2 both
            #     is_ty1_trans = 1
            #     ty1_sum_succ += 1
            #     ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
            #     Q_len = max(0., Q_len + ty2_gene_rate - (I - self.ty1_gene_rate))
            # else:                   # transfer ty2 only
            #     is_ty1_trans = 0
            #     ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
            #     Q_len = max(0., Q_len + ty2_gene_rate - I)

            energy_effi = sum_r / sum_p

            reward = self.get_reward(
                is_ty1_trans, ty1_succ_rate, Q_len, energy_effi_previous, 
                energy_effi, p)
        else:                       # transfer ty2 only
            self.situation_3 += 1
            is_ty1_trans = 0
            ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
            energy_effi_previous = sum_r / sum_p
            sum_r += min(I, Q_len)
            sum_p += p
            energy_effi = sum_r / sum_p
            Q_len = max(0., Q_len - I) + ty2_gene_rate

            reward = self.get_reward(
                is_ty1_trans, ty1_succ_rate, Q_len, energy_effi_previous, 
                energy_effi, p)

        return reward, ty1_sum_succ, Q_len, sum_r, sum_p, energy_effi

    def step(self, a_index):
        """
        do action, and ruturn reward
        """
        self.step_now += 1
        self.step_until_now += 1
        # p1, p2, encode = action
        p1, p2 = self.action[a_index], self.action[a_index]
        # encode is the index (0, 1, 2, 3) of action_discre list: [[1, 1], [1, 0], [0, 1], [0, 0]], 
        # '1' means transmit ty1, '0' means do not transmit ty1
        # if encode == 0:
        #     trans_ty1_1, trans_ty1_2 = True, True
        # elif encode == 1:
        #     trans_ty1_1, trans_ty1_2 = True, False
        # elif encode == 2:
        #     trans_ty1_1, trans_ty1_2 = False, True
        # else:
        #     trans_ty1_1, trans_ty1_2 = False, False

        # p1 = p1 / self.range * (self.power_max / 2) \
        #     + (self.power_max / 2) + 1e-15          # [-1, 1] -> dBm
        # p2 = p2 / self.range * (self.power_max / 2) \
        #     + (self.power_max / 2) + 1e-15
        # r1 = r1 / self.range * (self.rate_max / 2) \
        #   + (self.rate_max / 2) + 1e-15          # [-1, 1] -> kbps
        # r2 = r2 / self.range * (self.rate_max / 2) \
        #   + (self.rate_max / 2) + 1e-15

        self.ty2_gene_rate_1 = self.np_random.poisson(self.ty2_lamda)
        self.ty2_gene_rate_2 = self.np_random.poisson(self.ty2_lamda)

        h11, h22, h12, h21 = self.H
        p1_w = p1
        p2_w = p2
        self.I1 = self.band_width * np.log2(
            1 + h11 * p1_w / (self.N0 + h12 * p2_w))
        self.I2 = self.band_width * np.log2(
            1 + h22 * p2_w / (self.N0 + h21 * p1_w))

        # if self.step_until_now % self.steps_for_plot == 0:
        #     # self.all_actions.append([p1, p2])
        #     print('power and channel capacity:', p1, p2, self.I1, self.I2)

        reward_1, self.ty1_sum_succ_1, self.Q_len_1, self.sum_r_1, \
            self.sum_p_1, self.energy_effi_1 \
            = self.get_feedback(
                p1_w, 0, self.I1, self.ty2_gene_rate_1, 
                self.ty1_sum_succ_1, self.Q_len_1, self.sum_r_1, self.sum_p_1)

        reward_2, self.ty1_sum_succ_2, self.Q_len_2, self.sum_r_2, \
            self.sum_p_2, self.energy_effi_2 \
            = self.get_feedback(
                p2_w, 0, self.I2, self.ty2_gene_rate_2, 
                self.ty1_sum_succ_2, self.Q_len_2, self.sum_r_2, self.sum_p_2)

        if self.Q_len_1 > self.Q_max:
            self.large_than_Q_1 += 1
        if self.Q_len_2 > self.Q_max:
            self.large_than_Q_2 += 1

        self.H = self.get_channel_h()
        h11, h22, h12, h21 = self.H

        next_state = np.array([
            # self.ty1_sum_succ_1 / self.step_in_one_episode * self.s_amp,
            # self.ty1_sum_succ_2 / self.step_in_one_episode * self.s_amp, 
            self.Q_len_1 / self.Q_shr, self.Q_len_2 / self.Q_shr, 
            h11 * self.h_amp_self, h22 * self.h_amp_self, 
            h12 * self.h_amp_betw, h21 * self.h_amp_betw,
            # self.step_now / self.step_in_one_episode * 10
            ])
        # TODO: 增加额外信息如type1、2产生速率，发射功率

        reward = (reward_1 + reward_2) / 2

        done = False
        info = {}
        if self.step_now == self.step_in_one_episode:
            done = True
            info = {
                'ty1_succ_rate_1': self.ty1_sum_succ_1 / self.step_in_one_episode,
                'ty1_succ_rate_2': self.ty1_sum_succ_2 / self.step_in_one_episode,
                'Q_len_1': self.Q_len_1,
                'Q_len_2': self.Q_len_2,
                'energy_effi_1': self.energy_effi_1,
                'energy_effi_2': self.energy_effi_2, 
                'avg_rate': (self.sum_r_1 + self.sum_r_2) / self.step_in_one_episode / 2,
                'avg_power': (self.sum_p_1 + self.sum_p_2) / self.step_in_one_episode / 2, 
                # 10 * np.log10(1e3*(self.sum_p_1 + self.sum_p_2) / (self.step_in_one_episode+1) / 2), 
                }

        return next_state, reward, done, info

    def reset(self):
        """
        obtain initial parameters
        """
        self.step_now = 0

        self.ty1_sum_succ_1 = 0
        self.ty1_sum_succ_2 = 0
        self.Q_len_1 = 0  # self.np_random.uniform(low=0, high=self.Q_max)
        self.Q_len_2 = 0  # self.np_random.uniform(low=0, high=self.Q_max)

        self.H = self.get_channel_h()
        h11, h22, h12, h21 = self.H

        state = np.array([
            # self.ty1_sum_succ_1 / self.step_in_one_episode * self.s_amp, 
            # self.ty1_sum_succ_2 / self.step_in_one_episode * self.s_amp,
            self.Q_len_1 / self.Q_shr, self.Q_len_2 / self.Q_shr, 
            h11 * self.h_amp_self, h22 * self.h_amp_self, 
            h12 * self.h_amp_betw, h21 * self.h_amp_betw,
            # self.step_now / self.step_in_one_episode * 10
            ])

        # TODO: energy efficiency should be initialize properly
        self.sum_r_1 = 0
        self.sum_p_1 = 1e-8
        self.sum_r_2 = 0
        self.sum_p_2 = 1e-8

        return state


if __name__ == '__main__':
    env = EnvTwoUsers(30)
    env.seed(0)
    print(env.reset())

    power = [0, 1, 3, 5, 6]
    for p in power:
        print(env.step(p))
