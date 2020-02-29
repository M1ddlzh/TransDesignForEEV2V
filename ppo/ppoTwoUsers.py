"""
tensorflow r1.12
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import scipy.signal
from EnvTwoUsers import EnvTwoUsers
import os

tf.set_random_seed(0)
np.random.seed(0)

EP_MAX = 300   # 一共进行多少回合
EP_LEN = 1000    # 一回合多少步
GAMMA = 0.95
GAE_LAM = 0.95
A_LR = 0.00001
C_LR = 0.00001
BATCH = 128      # 对这些数据计算优势函数，更新PPO
A_UPDATE_STEPS = 10     # 每一组数据更新多次
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 10, 4    # 状态、动作维度
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   
    # KL penalty, kl_target是期望的目标,lam是KL散度在损失函数中的权重
    dict(name='clip', epsilon=0.1, max_lim=3000, kl_target=0.01),                 
    # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):
    """
    PPO Agent
    """

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            for _ in range(3):
                l1 = tf.layers.dense(l1, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, A_DIM)
            self.tfdc_r = tf.placeholder(tf.float32, [None, A_DIM], 'discounted_r')
            self.delta = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.delta))   
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('name_pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('name_oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample([1]), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, A_DIM], 'advantage')
        with tf.variable_scope('loss'): 
            with tf.variable_scope('surrogate'):
                # self.ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                self.ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-15)
                surr = self.ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better, 增加腾讯绝悟，dual-clipped ppo 
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -tf.reduce_mean(
                    tf.where(tf.less(self.tfadv, 0.),     # 动作变成4维了，这里的形状也得变
                    tf.maximum(
                    tf.minimum(surr,
                    tf.clip_by_value(self.ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv),
                    METHOD['max_lim'] * self.tfadv),
                    tf.minimum(surr,
                    tf.clip_by_value(self.ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv)))     

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r, adv):
        """
        update ppo agent
        """
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.delta, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this is in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean], 
                    {self.tfs: s, self.tfa: a, self.tfadv: adv})
                if kl > 2 * METHOD['kl_target']:  
                    break
            # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    # actor net
    def _build_anet(self, name, trainable):
        """
        build ppo agent network
        """
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.tanh, trainable=trainable)
            for _ in range(3):
                l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            # out1 = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
            # out2 = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
            # out3 = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
            # out4 = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
            # mu = tf.concat([out1, out2, out3, out4], 1)
            # out5 = tf.layers.dense(l1, 1, tf.nn.softplus, trainable=trainable)
            # out6 = tf.layers.dense(l1, 1, tf.nn.softplus, trainable=trainable)
            # out7 = tf.layers.dense(l1, 1, tf.nn.softplus, trainable=trainable)
            # out8 = tf.layers.dense(l1, 1, tf.nn.softplus, trainable=trainable)
            # sigma = tf.concat([out5, out6, out7, out8], 1)
            mu = tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # kernel_initializer=self.ortho_init(scale=1.), 
    def ortho_init(self, scale=1.0):
        """
        orthogonal initializer, used for network initialization.
        """
        def _ortho_init(shape, dtype, partition_info=None):
            #lasagne ortho init for tf
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4: # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v # pick the one with the correct shape
            q = q.reshape(shape)
            return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
        return _ortho_init

    def choose_action(self, s):
        """
        choose an action
        """
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]))

    def get_v(self, s):
        """
        get the value of state s
        """
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = EnvTwoUsers()

ppo = PPO()

total_reward = []
total_reward_not_smooth = []

total_type1_success_1 = []
total_type1_success_not_smooth_1 = []
total_queue_len_1 = []
total_queue_len_not_smooth_1 = []
total_energy_effi_1 = []
total_energy_effi_not_smooth_1 = []

total_type1_success_2 = []
total_type1_success_not_smooth_2 = []
total_queue_len_2 = []
total_queue_len_not_smooth_2 = []
total_energy_effi_2 = []
total_energy_effi_not_smooth_2 = []

power_max = env.power_max
power_min = env.power_min
rate_max = env.rate_max
rate_min = env.rate_min
all_power_1 = []
all_rate_1 = []
all_power_2 = []
all_rate_2 = []
all_power_1_smooth = []
all_rate_1_smooth = []
all_power_2_smooth = []
all_rate_2_smooth = []

channel_capacity_1 = []
channel_capacity_2 = []
channel_capacity_1_smooth = []
channel_capacity_2_smooth = []


for ep in tqdm(range(EP_MAX), ncols = 50):
    s, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
        Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2 = env.reset()      # TODO:Q目前是小数
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0.
    all_r = []
    for t in range(EP_LEN):    # in one episode
        action = ppo.choose_action(s)    # p1, p2, r1, r2 = a
        s_, r, done, H_, \
        ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
        Q_len_1, Q_len_2, \
        sum_r_1, sum_p_1, sum_r_2, sum_p_2, energy_effi_1, energy_effi_2, I1, I2 = \
            env.step([action, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, 
                     Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2])    
        buffer_s.append(s)
        buffer_a.append(action)
        buffer_r.append((r - 11) / 8)    # TODO: 确定reward的均值和方差, ee:-50, ts:+9, q:-50
        ep_r += r
        all_r.append(r)

        if (EP_MAX - ep) < 5:
            a = action.copy()
            a[0] = a[0] * (power_max / 2) + (power_max / 2) + 1e-15       # dBm
            a[1] = a[1] * (power_max / 2) + (power_max / 2) + 1e-15
            a[2] = a[2] * (rate_max / 2) + (rate_max / 2) + 1e-15
            a[3] = a[3] * (rate_max / 2) + (rate_max / 2) + 1e-15
            all_power_1.append(a[0]); all_power_2.append(a[1])
            all_rate_1.append(a[2]); all_rate_2.append(a[3])
            channel_capacity_1.append(I1); channel_capacity_2.append(I2)
            if t == 0:
                all_power_1_smooth.append(a[0]); all_power_2_smooth.append(a[1]) 
                all_rate_1_smooth.append(a[2]); all_rate_2_smooth.append(a[3])
                channel_capacity_1_smooth.append(I1); channel_capacity_2_smooth.append(I2)
            else:
                all_power_1_smooth.append(all_power_1_smooth[-1] * 0.9 + 0.1 * a[0])
                all_power_2_smooth.append(all_power_2_smooth[-1] * 0.9 + 0.1 * a[1])
                all_rate_1_smooth.append(all_rate_1_smooth[-1] * 0.9 + 0.1 * a[2])
                all_rate_2_smooth.append(all_rate_2_smooth[-1] * 0.9 + 0.1 * a[3])
                channel_capacity_1_smooth.append(channel_capacity_1_smooth[-1] * 0.9 + 0.1 * I1)
                channel_capacity_2_smooth.append(channel_capacity_2_smooth[-1] * 0.9 + 0.1 * I2)

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            discounted_r = []
            gae = []
            v_s_ = ppo.get_v(s_)
            v_s_ = (r-10)/8 + GAMMA * v_s_
            discounted_r.append(v_s_)
            v_s = ppo.get_v(s)
            gae_delta = v_s_ - v_s
            gae.append(gae_delta)
            for r, s, st_ in zip(buffer_r[:-1][::-1], buffer_s[:-1][::-1], buffer_s[1:][::-1]):
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
                v_s = ppo.get_v(s)
                gae_v_s_ = r + GAMMA * ppo.get_v(st_)
                gae_delta = gae_v_s_ - v_s
                gae.append(gae_delta)
            discounted_r.reverse()
            gae.reverse()
            # GAE, two implementation methods
            gae = scipy.signal.lfilter([1], [1, float(-GAMMA * GAE_LAM)], gae[::-1], axis=0)[::-1]
            # for t in reversed(range(len(gae) - 1)):
            #     gae[t] = gae[t] + GAMMA * GAE_LAM * gae[t + 1]
            discounted_r = np.array(discounted_r)[:, np.newaxis]
            gae = np.array(gae)[:, np.newaxis]
            bs, ba, br, badv_gae = np.vstack(buffer_s), np.vstack(buffer_a), \
                np.hstack((discounted_r, discounted_r, discounted_r, discounted_r)), \
                np.hstack((gae, gae, gae, gae))         # 重复4次是因为 A_DIM = 4
            ppo.update(bs, ba, br, badv_gae)
            buffer_s, buffer_a, buffer_r = [], [], []
        s = s_
        H = H_
    # print(np.array(all_r).mean(), np.array(all_r).std())

    # 画图数据
    if ep == 0: 
        total_reward.append(ep_r)
        total_type1_success_1.append(ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1))
        total_type1_success_2.append(ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2))
        total_queue_len_1.append(Q_len_1)
        total_queue_len_2.append(Q_len_2)
        total_energy_effi_1.append(energy_effi_1)
        total_energy_effi_2.append(energy_effi_2)
    else: 
        total_reward.append(total_reward[-1]*0.9 + ep_r*0.1)  # 就是Tensorboard中的平滑函数，一阶IIR滤波器
        total_type1_success_1.append(total_type1_success_1[-1]*0.9 + 
            ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1) * 0.1)
        total_type1_success_2.append(total_type1_success_2[-1]*0.9 +
            ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2) * 0.1)
        total_queue_len_1.append(total_queue_len_1[-1]*0.9 + Q_len_1*0.1)
        total_queue_len_2.append(total_queue_len_2[-1]*0.9 + Q_len_2*0.1)
        total_energy_effi_1.append(total_energy_effi_1[-1]*0.9 + energy_effi_1*0.1)
        total_energy_effi_2.append(total_energy_effi_2[-1]*0.9 + energy_effi_2*0.1)
    total_reward_not_smooth.append(ep_r)
    total_type1_success_not_smooth_1.append(ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1))
    total_queue_len_not_smooth_1.append(Q_len_1)
    total_energy_effi_not_smooth_1.append(energy_effi_1)
    total_type1_success_not_smooth_2.append(ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2))
    total_queue_len_not_smooth_2.append(Q_len_2)
    total_energy_effi_not_smooth_2.append(energy_effi_2)


fig, axs = plt.subplots(2, 2, constrained_layout=True)
total_fig = ['total_reward', 'total_type1_success', 'total_queue_len', 'total_energy_effi']
fig_i = 0
for ax in axs.flatten():
    if fig_i == 0:
        ax.plot(eval(total_fig[fig_i] + '_not_smooth'), color = 'bisque')
        ax.plot(eval(total_fig[fig_i]), color = 'darkorange')
        ax.set_title(total_fig[fig_i].replace('_', ' '))
    else:
        ax.plot(eval(total_fig[fig_i] + '_not_smooth_1'), color = 'bisque')
        ax.plot(eval(total_fig[fig_i] + '_not_smooth_2'), color = 'lightblue')
        ax.plot(eval(total_fig[fig_i] + '_1'), color = 'darkorange')
        ax.plot(eval(total_fig[fig_i] + '_2'), color = 'royalblue')
        ax.set_title(total_fig[fig_i].replace('_', ' '))
    ax.grid(True)
    fig_i += 1
plt.suptitle('training epoch = ' + str(EP_MAX) + ' training result')

path = './fig_result/'
if not os.path.exists(path):
    os.makedirs(path)

plt.savefig(path + str(EP_MAX) + ' train_result_' + 
            datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)
# plt.show()

# plot power and rate
fig, axs = plt.subplots(1, 2, constrained_layout=True)
fig_i = 1
for ax in axs.flatten():
    if fig_i == 1:
        ax.plot(eval('all_power_1'), color = 'bisque')
        ax.plot(eval('all_power_2'), color = 'lightblue')
        ax.plot(eval('all_power_1' + '_smooth'), color = 'darkorange')
        ax.plot(eval('all_power_2' + '_smooth'), color = 'royalblue')
        ax.set_title('Power')
        ax.grid(True)
    else:
        ax.plot(eval('all_rate_1'), color = 'bisque')
        ax.plot(eval('all_rate_2'), color = 'lightblue')
        ax.plot(eval('all_rate_1' + '_smooth'), color = 'darkorange')
        ax.plot(eval('all_rate_2' + '_smooth'), color = 'royalblue')
        ax.set_title('Rate')
        ax.grid(True)
    fig_i += 1
plt.suptitle('training epoch = ' + str(EP_MAX) + ' training result')
plt.savefig(path + str(EP_MAX) + ' action_' + 
            datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

# plot channel capacity
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(channel_capacity_1, color = 'bisque')
axs.plot(channel_capacity_2, color = 'lightblue')
axs.plot(channel_capacity_1_smooth, color = 'darkorange')
axs.plot(channel_capacity_2_smooth, color = 'royalblue')
axs.set_title('Channel Capacity')
axs.grid(True)
plt.suptitle('training epoch = ' + str(EP_MAX) + ' training result')
plt.savefig(path + str(EP_MAX) + ' channel_capacity_' + 
            datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)

plt.show()


r"""
env = EnvTwoUsers()

total_reward = []
total_reward_not_smooth = []

total_type1_success_1 = []
total_type1_success_not_smooth_1 = []
total_queue_len_1 = []
total_queue_len_not_smooth_1 = []
total_energy_effi_1 = []
total_energy_effi_not_smooth_1 = []

total_type1_success_2 = []
total_type1_success_not_smooth_2 = []
total_queue_len_2 = []
total_queue_len_not_smooth_2 = []
total_energy_effi_2 = []
total_energy_effi_not_smooth_2 = []

for ep in tqdm(range(100), ncols = 50):
    s, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
        Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2 = env.reset()
    ep_r = 0.
    for _ in range(EP_LEN):
        a = ppo.choose_action(s)
        s_, r, done, H_, \
        ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
        Q_len_1, Q_len_2, \
        sum_r_1, sum_p_1, sum_r_2, sum_p_2, energy_effi_1, energy_effi_2 = \
            env.step([a, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, 
                      Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2])
        H = H_
        s = s_
        ep_r += r

    if ep == 0: 
        total_reward.append(ep_r)
        total_type1_success_1.append(ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1))
        total_type1_success_2.append(ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2))
        total_queue_len_1.append(Q_len_1)
        total_queue_len_2.append(Q_len_2)
        total_energy_effi_1.append(energy_effi_1)
        total_energy_effi_2.append(energy_effi_2)
    else: 
        total_reward.append(total_reward[-1]*0.9 + ep_r*0.1)  # 就是Tensorboard中的平滑函数，一阶IIR滤波器
        total_type1_success_1.append(total_type1_success_1[-1]*0.9 + 
            ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1) * 0.1)
        total_type1_success_2.append(total_type1_success_2[-1]*0.9 +
            ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2) * 0.1)
        total_queue_len_1.append(total_queue_len_1[-1]*0.9 + Q_len_1*0.1)
        total_queue_len_2.append(total_queue_len_2[-1]*0.9 + Q_len_2*0.1)
        total_energy_effi_1.append(total_energy_effi_1[-1]*0.9 + energy_effi_1*0.1)
        total_energy_effi_2.append(total_energy_effi_2[-1]*0.9 + energy_effi_2*0.1)
    total_reward_not_smooth.append(ep_r)
    total_type1_success_not_smooth_1.append(ty1_sum_succ_1 / (ty1_sum_succ_1 + ty1_sum_fail_1))
    total_queue_len_not_smooth_1.append(Q_len_1)
    total_energy_effi_not_smooth_1.append(energy_effi_1)
    total_type1_success_not_smooth_2.append(ty1_sum_succ_2 / (ty1_sum_succ_2 + ty1_sum_fail_2))
    total_queue_len_not_smooth_2.append(Q_len_2)
    total_energy_effi_not_smooth_2.append(energy_effi_2)

fig, axs = plt.subplots(2, 2, constrained_layout=True)
total_fig = ['total_reward', 'total_type1_success', 'total_queue_len', 'total_energy_effi']
fig_i = 0
for ax in axs.flatten():
    if fig_i == 0:
        ax.plot(eval(total_fig[fig_i] + '_not_smooth'), color = 'bisque')
        ax.plot(eval(total_fig[fig_i]), color = 'darkorange')
        ax.set_title(total_fig[fig_i].replace('_', ' '))
    else:
        ax.plot(eval(total_fig[fig_i] + '_not_smooth_1'), color = 'bisque')
        ax.plot(eval(total_fig[fig_i] + '_not_smooth_2'), color = 'lightblue')
        ax.plot(eval(total_fig[fig_i] + '_1'), color = 'darkorange')
        ax.plot(eval(total_fig[fig_i] + '_2'), color = 'royalblue')
        ax.set_title(total_fig[fig_i].replace('_', ' '))
    ax.grid(True)
    fig_i += 1
plt.suptitle('testing epoch = ' + str(100) + ' testing result')

plt.savefig(path + str(EP_MAX) + ' test_result_' + 
            datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)
plt.show()
"""


"""
验证腾讯dual-clipped PPO的优势：影响不大，3，甚至有反作用：轻微震荡
讨论clip超参影响：影响不大，0.2/0.1
加上GAE：效果很差
奖励r的标准化：很重要
advantage的标准化：不明确，甚至有反作用：性能很差
正交初始化：作用不大
增大batch size
第一层用tanh激活
"""

"""
对比算法：DQN，PPO，Lyapunov优化；TODO: 模仿TCP/IP的流量控制，随机算法的好坏太依赖power和rate的取值范围
对比思路：ty1和ty2数据量增大；衰落因子m变化；车距变化；3对用户
离散动作：spinningup-ppo-core.py, log probability
reward normalization
增大ty1成功率和能效比会下降吗的关系
"""