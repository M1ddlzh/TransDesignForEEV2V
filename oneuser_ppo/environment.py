import numpy as np
import math

np.random.seed(0)

sum_P = 0
sum_R = 0
sum_phi = 0
reward = 0

alpha = 0.5  # 能效
beta = 1  # 成功率 1.3 1.2265
gamma = 1  # 队列长度 1.35 1.4

class Feedback:
    """
    single users pair environment
    """
    def __init__(self,
                 r,
                 A_t,
                 H_t,
                 N0,
                 phi_min,
                 phi_0,
                 Q_max,
                 Q_limit,
                 Q_infi,
                 K_1,
                 K_2,
                 K_3,
                 K_4,
                 n_ts,
                 n_q,
                 one_train
                 ):
        super(Feedback, self).__init__()
        self.r = r
        self.A_t = A_t
        self.H_t = H_t

        self.N0 = N0
        self.phi_min = phi_min
        self.phi_0 = phi_0
        self.Q_max = Q_max
        self.Q_limit = Q_limit
        self.Q_infi = Q_infi
        self.K_1 = K_1
        self.K_2 = K_2
        self.K_3 = K_3
        self.K_4 = K_4
        self.n_ts = n_ts
        self.n_q = n_q
        self.one_train = one_train

        self.num_h = len(H_t)

        self.rate_min = 8.          # band width = 10 kHz, kbps
        self.rate_max = 16.  
        self.power_min = 0.17       # w
        self.power_max = 2.05
    
    def get_reward(self, sigma_m_nt, r_nt, sum_R_nt, sum_P_nt, p_nt, 
                    s_m_1_nt, s_m_2_t):
        ######################################################################
        # 能效比奖励归一化
        r_ee = sigma_m_nt * r_nt - sum_R_nt / sum_P_nt * p_nt

        r_ee_min = - self.power_max
        r_ee_max = self.rate_max
        if r_ee >= 0:
            r_ee_norma = r_ee / r_ee_max
        else:
            r_ee_norma = - r_ee / r_ee_min

        ######################################################################
        # type 1成功率奖励归一化
        if s_m_1_nt < self.phi_min:
            r_ts = - 100
        elif self.phi_min <= s_m_1_nt < self.phi_0:
            r_ts = -80
        else:
            r_ts = 0.5 * s_m_1_nt + 95

        r_ts_max = 100
        r_ts_min = -100
        if r_ts >= 0:
            r_ts_norma = r_ts / r_ts_max
        else:
            r_ts_norma = - r_ts / r_ts_min

        ######################################################################
        # 队列奖励归一化
        if s_m_2_t < self.Q_max:
            r_q = -0.5 * s_m_2_t + 10
        elif self.Q_max < s_m_2_t < self.Q_infi:
            r_q = -s_m_2_t + 20
        else:
            r_q = -40
        
        if r_q >= 0:
            r_q_norma = r_q / 10
        else:
            r_q_norma = r_q / 40
        # print('Reward q length', r_q)
        ######################################################################

        reward = alpha * r_ee_norma + beta * r_ts_norma + gamma * r_q_norma
        # print('Sub reward', alpha * r_ee_norma, beta * r_ts_norma, gamma * r_q_norma)
        # print('Sub reward', alpha * r_ee, beta * r_ts, gamma * r_q)
        # print('Total reward', reward)
        
        return reward

    def get_env_feedback(self, s, a, Q_pt, num_slot):
        global sum_P, sum_R, sum_phi, reward
        # print("sum_P_t:", sum_P)
        # print("sum_R_t:", sum_R)

        current = s
        s_m_2_t = current[1]            # max(Q_pt - b_t, 0) + a_t
        h_t = current[2]                # h_pt

        next_state = []
        Q_t = 0

        # cond_h = np.random.uniform()
        # print("cond_h:", cond_h)
        if num_slot != self.one_train:
            bad_condition = False
            if h_t == self.H_t[0]:
                p_1i = np.array([0.9, 0.1])
                h_nt = np.random.choice([self.H_t[0], self.H_t[1]], p=p_1i)
            elif h_t == self.H_t[self.num_h - 1]:
                p_10i = np.array([0.1, 0.9])
                h_nt = np.random.choice([self.H_t[self.num_h - 2], self.H_t[self.num_h - 1]], p=p_10i)
            else:
                index = 0
                for i in range(len(self.H_t)):
                    if h_t == self.H_t[i]:
                        index = i
                p_ji = np.array([0.2, 0.6, 0.2])
                h_nt = np.random.choice([self.H_t[index - 1], self.H_t[index], self.H_t[index + 1]], p=p_ji)
        else:
            h_nt = 0
            bad_condition = True

        p_nt = a[0]
        r_nt = a[1]
        p_nt = p_nt * (self.power_max / 2) + (self.power_max / 2) + 1e-15       # [-1, 1] -> Watt
        r_nt = r_nt * (self.rate_max / 2) + (self.rate_max / 2) + 1e-15

        # if r_nt_type1 == 10:    # TODO: 当传输速率小于信道容量且大于ty1速率时，是否一定传输ty1，225行
        #     sigma_nt = 1
        # else:
        #     sigma_nt = 2
        sigma_nt = 1

        sum_P += p_nt
        sum_P_nt = sum_P

        i_nt = 10 * math.log((1 + p_nt * h_t ** 2 / self.N0), 2)    # band width = 10 kHz, I -> kbps, 14~42kbps
        # print("I_t", i_t)
        
        a_nt = np.random.choice(self.A_t)   # type2产生速率

        # transfer failed
        if r_nt > i_nt:
            # print("r_t > i_t")
            sum_phi += 0
            sum_phi_nt = sum_phi
            avrg_phi_nt = round(sum_phi_nt / num_slot, 4)
            # print(avrg_phi_nt)

            s_m_1_nt = avrg_phi_nt

            b_nt = 0
            s_m_2_nt = max(Q_pt - b_nt, 0) + a_nt
            
            # print("b_t", b_nt)
            # print("a_t", a_nt)
            """
            if s_m_2_nt > self.Q_max:
                s_m_2_nt = self.Q_max + 100
            """
            
            Q_t = s_m_2_nt

            sigma_m_1_nt = 0
            sigma_m_2_nt = 0
            next_state = [s_m_1_nt, s_m_2_nt, h_nt, sigma_m_1_nt, sigma_m_2_nt]

            sigma_m_nt = 0
            sum_R += sigma_m_nt * r_nt
            sum_R_nt = sum_R
        
            reward = self.get_reward(sigma_m_nt, r_nt, sum_R_nt, sum_P_nt, p_nt, 
                s_m_1_nt, s_m_2_t)

        # transfer ty2 only
        elif (r_nt <= i_nt) and (r_nt < self.r):
            # print("r_t<=i_t, r_t<r")
            sum_phi += 0
            sum_phi_nt = sum_phi
            avrg_phi_nt = round(sum_phi_nt / num_slot, 4)
            # print(avrg_phi_nt)

            s_m_1_nt = avrg_phi_nt

            b_nt = r_nt
            s_m_2_nt = max(Q_pt - b_nt, 0) + a_nt
            Q_t = s_m_2_nt

            sigma_m_1_nt = 0
            sigma_m_2_nt = 1
            next_state = [s_m_1_nt, s_m_2_nt, h_nt, sigma_m_1_nt, sigma_m_2_nt]

            sigma_m_nt = 1
            sum_R += sigma_m_nt * r_nt
            sum_R_nt = sum_R

            reward = self.get_reward(sigma_m_nt, r_nt, sum_R_nt, sum_P_nt, p_nt, 
                s_m_1_nt, s_m_2_t)

        elif self.r <= r_nt <= i_nt:
            # transfer ty1 and ty2 both
            if sigma_nt == 1:
                # print("r <= r_t <= i_t, sigma_t = 1")
                sum_phi += 1
                sum_phi_nt = sum_phi
                avrg_phi_nt = round(sum_phi_nt / num_slot, 4)
                # print(avrg_phi_nt)

                s_m_1_nt = avrg_phi_nt

                b_nt = r_nt - self.r
                s_m_2_nt = max(Q_pt - b_nt, 0) + a_nt
                Q_t = s_m_2_nt

                sigma_m_1_nt = 1
                sigma_m_2_nt = 1
                next_state = [s_m_1_nt, s_m_2_nt, h_nt, sigma_m_1_nt, sigma_m_2_nt]

                sigma_m_nt = 1
                sum_R += sigma_m_nt * r_nt
                sum_R_nt = sum_R

                reward = self.get_reward(sigma_m_nt, r_nt, sum_R_nt, sum_P_nt, p_nt, 
                    s_m_1_nt, s_m_2_t)

            # still transfer ty2 only
            elif sigma_nt == 2:
                # print("r <= r_t <= i_t, sigma_t = 2")
                sum_phi += 0
                sum_phi_nt = sum_phi
                avrg_phi_nt = round(sum_phi_nt / num_slot, 4)
                # print(avrg_phi_nt)

                s_m_1_nt = avrg_phi_nt

                b_nt = r_nt
                s_m_2_nt = max(Q_pt - b_nt, 0) + a_nt
                Q_t = s_m_2_nt

                sigma_m_1_nt = 0
                sigma_m_2_nt = 1
                next_state = [s_m_1_nt, s_m_2_nt, h_nt, sigma_m_1_nt, sigma_m_2_nt]

                sigma_m_nt = 1
                sum_R += sigma_m_nt * r_nt
                sum_R_nt = sum_R

                reward = self.get_reward(sigma_m_nt, r_nt, sum_R_nt, sum_P_nt, p_nt, 
                    s_m_1_nt, s_m_2_t)

        return next_state, reward, bad_condition, Q_t, sum_R, sum_P

    def reset(self):
        global sum_P, sum_R, sum_phi, reward
        sum_P = 0
        sum_R = 0
        sum_phi = 0
        reward = 0

