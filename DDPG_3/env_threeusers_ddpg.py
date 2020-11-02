import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class EnvThreeUsers(gym.Env):
    """
    Three Users Env
    """
    def __init__(self, steps_in_one_episode):
        self.step_in_one_episode = steps_in_one_episode

        self.seed()
        self.step_now = 0
        ############### channel parameters ###############
        self.N0 = 1
        self.band_width = 1
        self.scale_ii = np.sqrt(1 / 2)
        self.scale_ij = np.sqrt(0.03 / 2)
        self.scale_ij_2 = np.sqrt(0.001 / 2)

        ############### amount of date parameters ###############
        self.ty1_gene_rate = 0.5        # type1 generate rate, kbps
        self.ty2_lamda = 1              # type2 poission mean, kbps 

        ############### reward function parameters ###############
        self.phi_0 = 1            
        self.Q_max = 10 * self.ty2_lamda
        self.alpha = 100                # ee reward weight 
        self.beta = 10                  # ty1 reward weight 
        self.gamma = 2                  # queue length weight 

        ############### simple state features scaling parameters ###########
        self.s_amp = 10
        self.Q_shr = self.ty2_lamda
        self.h_amp_self = 1
        self.h_amp_betw = 100
        self.h_amp_betw_2 = 1000

        ############### counter of rate and channle capacity ###############
        self.situation_1 = 0
        self.situation_2 = 0
        self.situation_3 = 0
        self.large_than_Q_1 = 0
        self.large_than_Q_2 = 0
        self.large_than_Q_3 = 0

        ############### action and state parameters ###############
        self.steps_for_plot = 40000
        self.step_until_now = 0
        self.all_actions = []
        self.range = 10
        self.power_min = 0
        self.power_max = 100
        self.b = (self.power_min + self.power_max) / 2
        self.k = (self.power_max - self.b) / self.range
        self.a_min = -self.range
        self.a_max = self.range
        self.action_min = np.array([self.a_min]*3, dtype=np.float32)
        self.action_max = np.array([self.a_max]*3, dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.action_min, high=self.action_max)
        self.observation_space = spaces.Box(
            low=0., high=float("inf"), shape=(self.reset().shape[0],), dtype=np.float32)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_channel_h(self):
        """
        get channel parameters
        """
        h11 = (self.np_random.rayleigh(self.scale_ii)) ** 2
        h22 = (self.np_random.rayleigh(self.scale_ii)) ** 2
        h33 = (self.np_random.rayleigh(self.scale_ii)) ** 2
        h12 = (self.np_random.rayleigh(self.scale_ij)) ** 2
        h21 = (self.np_random.rayleigh(self.scale_ij)) ** 2
        h13 = (self.np_random.rayleigh(self.scale_ij_2)) ** 2
        h31 = (self.np_random.rayleigh(self.scale_ij_2)) ** 2
        h32 = (self.np_random.rayleigh(self.scale_ij)) ** 2
        h23 = (self.np_random.rayleigh(self.scale_ij)) ** 2
        H = [h11, h22, h33, h12, h21, h13, h31, h32, h23]
        return H

    def get_reward(self, is_ty1_trans, ty1_succ_rate, Q_len, 
        energy_effi_previous, energy_effi, p):
        """
        get reward
        """
        # energy efficiency reward 
        r_ee = energy_effi - energy_effi_previous

        # type1 reward
        if ty1_succ_rate < self.phi_0 + 0.01:
            r_ts = 2 * is_ty1_trans - 1
        else:
            r_ts = 0

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
        three cases of transmission
        """
        if self.ty1_gene_rate <= I:
            energy_effi_previous = sum_r / sum_p
            sum_r += min(I, self.ty1_gene_rate + Q_len)
            sum_p += p
            self.situation_1 += 1
            if ty1_sum_succ / self.step_in_one_episode < self.phi_0 + 0.01:
                is_ty1_trans = 1
                ty1_sum_succ += 1
                ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
                Q_len = max(
                    0., Q_len - (I - self.ty1_gene_rate)) + ty2_gene_rate
            else:
                is_ty1_trans = 0
                ty1_succ_rate = ty1_sum_succ / self.step_in_one_episode
                Q_len = max(0., Q_len - I) + ty2_gene_rate
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

    def step(self, action):
        """
        do action, and ruturn reward
        """
        self.step_now += 1
        self.step_until_now += 1
        p1, p2, p3 = action
        p1 = p1 * self.k + self.b
        p2 = p2 * self.k + self.b
        p3 = p3 * self.k + self.b
        p1_w = p1
        p2_w = p2
        p3_w = p3

        self.ty2_gene_rate_1 = self.np_random.poisson(self.ty2_lamda)
        self.ty2_gene_rate_2 = self.np_random.poisson(self.ty2_lamda)
        self.ty2_gene_rate_3 = self.np_random.poisson(self.ty2_lamda)

        h11, h22, h33, h12, h21, h13, h31, h32, h23 = self.H
        self.I1 = self.band_width * np.log2(
            1 + h11 * p1_w / (self.N0 + h12 * p2_w + h13 * p3_w))
        self.I2 = self.band_width * np.log2(
            1 + h22 * p2_w / (self.N0 + h21 * p1_w + h23 * p3_w))
        self.I3 = self.band_width * np.log2(
            1 + h33 * p3_w / (self.N0 + h31 * p1_w + h32 * p2_w))

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
        
        reward_3, self.ty1_sum_succ_3, self.Q_len_3, self.sum_r_3, \
            self.sum_p_3, self.energy_effi_3 \
            = self.get_feedback(
                p3_w, 0, self.I3, self.ty2_gene_rate_3, 
                self.ty1_sum_succ_3, self.Q_len_3, self.sum_r_3, self.sum_p_3)

        if self.Q_len_1 > self.Q_max:
            self.large_than_Q_1 += 1
        if self.Q_len_2 > self.Q_max:
            self.large_than_Q_2 += 1
        if self.Q_len_3 > self.Q_max:
            self.large_than_Q_3 += 1

        self.H = self.get_channel_h()
        h11, h22, h33, h12, h21, h13, h31, h32, h23 = self.H

        next_state = np.array([
            # self.ty1_sum_succ_1 / self.step_in_one_episode * self.s_amp,
            # self.ty1_sum_succ_2 / self.step_in_one_episode * self.s_amp, 
            # self.ty1_sum_succ_3 / self.step_in_one_episode * self.s_amp, 
            self.Q_len_1 / self.Q_shr, self.Q_len_2 / self.Q_shr, 
            self.Q_len_3 / self.Q_shr, 
            h11 * self.h_amp_self, h22 * self.h_amp_self, h33 * self.h_amp_self, 
            h12 * self.h_amp_betw, h21 * self.h_amp_betw,
            h13 * self.h_amp_betw_2, h31 * self.h_amp_betw_2,
            h32 * self.h_amp_betw, h23 * self.h_amp_betw,
            # self.step_now / self.step_in_one_episode * 10
            ])  

        reward = (reward_1 + reward_2 + reward_3) / 3

        done = False
        info = {}
        if self.step_now == self.step_in_one_episode:
            done = True
            info = {
                'ty1_succ_rate_1': self.ty1_sum_succ_1 / self.step_in_one_episode,
                'ty1_succ_rate_2': self.ty1_sum_succ_2 / self.step_in_one_episode,
                'ty1_succ_rate_3': self.ty1_sum_succ_3 / self.step_in_one_episode,
                'Q_len_1': self.Q_len_1,
                'Q_len_2': self.Q_len_2,
                'Q_len_3': self.Q_len_3,
                'energy_effi_1': self.energy_effi_1,
                'energy_effi_2': self.energy_effi_2, 
                'energy_effi_3': self.energy_effi_3, 
                'avg_rate': ((self.sum_r_1 + self.sum_r_2 + self.sum_r_3)
                    / self.step_in_one_episode / 3),
                'avg_power': ((self.sum_p_1 + self.sum_p_2 + self.sum_p_3) 
                    / self.step_in_one_episode / 3),
                }

        return next_state, reward, done, info

    def reset(self):
        """
        obtain initial parameters
        """
        self.step_now = 0

        self.ty1_sum_succ_1 = 0
        self.ty1_sum_succ_2 = 0
        self.ty1_sum_succ_3 = 0
        self.Q_len_1 = 0  # self.np_random.uniform(low=0, high=self.Q_max)
        self.Q_len_2 = 0  # self.np_random.uniform(low=0, high=self.Q_max)
        self.Q_len_3 = 0

        self.H = self.get_channel_h()
        h11, h22, h33, h12, h21, h13, h31, h32, h23 = self.H

        state = np.array([
            # self.ty1_sum_succ_1 / self.step_in_one_episode * self.s_amp,
            # self.ty1_sum_succ_2 / self.step_in_one_episode * self.s_amp, 
            # self.ty1_sum_succ_3 / self.step_in_one_episode * self.s_amp, 
            self.Q_len_1 / self.Q_shr, self.Q_len_2 / self.Q_shr, 
            self.Q_len_3 / self.Q_shr, 
            h11 * self.h_amp_self, h22 * self.h_amp_self, h33 * self.h_amp_self, 
            h12 * self.h_amp_betw, h21 * self.h_amp_betw,
            h13 * self.h_amp_betw_2, h31 * self.h_amp_betw_2,
            h32 * self.h_amp_betw, h23 * self.h_amp_betw,
            # self.step_now / self.step_in_one_episode * 10
            ])

        self.sum_r_1 = 0       
        self.sum_p_1 = 1e-8
        self.sum_r_2 = 0
        self.sum_p_2 = 1e-8
        self.sum_r_3 = 0
        self.sum_p_3 = 1e-8

        return state


if __name__ == '__main__':
    # testing
    env = EnvThreeUsers(30)
    env.seed(0)
    print(env.reset())

    power = [[9, 9, 0], [5, 5, 0], [0, 0, 0], [-5, -5, 0], [-9, -9, 0]]
    for p in power:
        print(env.step(p))
