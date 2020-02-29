import os
from environment import Feedback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import datetime
import scipy.signal

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
S_DIM, A_DIM = 5, 2    # 状态、动作维度
METHOD = [dict(name='kl_pen', kl_target=0.01, lam=0.5),   
          dict(name='clip', epsilon=0.1, max_lim=3000, kl_target=0.01)][1]       

ty1_gene = 10           # ty1 generate rate 
A_t = [5, 2, 4, 6, 1]   # ty2 generate rate 
H_t = []
while len(H_t) < 7:
    h_t = np.random.rayleigh(3 * 10 ** -7, 1)
    if h_t >= 2.5 * 10 ** -7 and h_t <= 4.5 * 10 ** -7:
        H_t.append(round(float(h_t), 10))
# print(H_t)
N0 = 10**-14
phi_min = 0.5
phi_0 = 0.80
Q_max = 20
Q_limit = 30
Q_infi = 40
K_1 = 5
K_2 = 5
K_3 = 5
K_4 = 5
n_ts = 10
n_q = 1.999

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
        return np.clip(a, np.array([-1, -1]), np.array([1, 1]))     # rely on A_DIM 

    def get_v(self, s):
        """
        get the value of state s
        """
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


######################################################################################################
ppo = PPO()
env = Feedback(r = ty1_gene, A_t = A_t, H_t = H_t, N0 = N0, 
               phi_min = phi_min, phi_0 = phi_0, Q_max = Q_max, Q_limit = Q_limit, Q_infi = Q_infi,
               K_1 = K_1, K_2 = K_2, K_3 = K_3, K_4 = K_4, n_ts = n_ts, n_q = n_q, one_train = EP_LEN)

total_reward_collection = []

for epoch in tqdm(range(EP_MAX), ncols = 50):
    slot = 0

    s_m_1_t0 = 0
    s_m_2_t0 = 0
    h_t0 = np.random.choice(H_t)
    sigma_m_1_t0 = 0
    sigma_m_2_t0 = 0
    Q_pt = 0
    s = [s_m_1_t0, s_m_2_t0, h_t0, sigma_m_1_t0, sigma_m_2_t0]  # initial state

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0.
    all_r = []

    while True:
        slot += 1
        action = ppo.choose_action(np.array(s))
        s_, reward, end, Q_, sum_R, sum_P = env.get_env_feedback(s, action, Q_pt, slot)     # s and s_ are list
        buffer_s.append(np.array(s))
        buffer_a.append(action)
        buffer_r.append((reward - 10) / 8)    # TODO: 确定reward的均值和方差
        ep_r += reward
        all_r.append(reward)
        
        # update ppo
        if slot % BATCH == 0 or end:
            discounted_r = []
            gae = []
            v_s_ = ppo.get_v(np.array(s_))
            v_s_ = (reward - 10) / 8 + GAMMA * v_s_
            discounted_r.append(v_s_)
            v_s = ppo.get_v(np.array(s))
            gae_delta = v_s_ - v_s
            gae.append(gae_delta)
            for reward, s, st_ in zip(buffer_r[:-1][::-1], buffer_s[:-1][::-1], buffer_s[1:][::-1]):
                v_s_ = reward + GAMMA * v_s_
                discounted_r.append(v_s_)
                v_s = ppo.get_v(s)
                gae_v_s_ = reward + GAMMA * ppo.get_v(st_)
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
                np.hstack((discounted_r, discounted_r)), \
                np.hstack((gae, gae))         # 重复2次是因为 A_DIM = 2
            ppo.update(bs, ba, br, badv_gae)
            buffer_s, buffer_a, buffer_r = [], [], []
        
        s = s_
        Q_pt = Q_

        if end:
            env.reset()
            break
    total_reward_collection.append(ep_r)

path = './singel_fig_result/'
if not os.path.exists(path):
    os.makedirs(path)

size = 15
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.plot(np.arange(len(total_reward_collection)), total_reward_collection)
plt.ylabel('Average Accumulated Q-value', fontsize = size)
plt.xlabel('Epoch', fontsize = size)
plt.savefig(path + 'Q_value_' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
plt.show()


