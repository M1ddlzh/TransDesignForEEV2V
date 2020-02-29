import os
from DQNEnvTwoUsers import DQNEnvTwoUsers
from DQNnet import DeepQNetwork
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime


EP_MAX = 100
EP_LEN = 1000 

not_learn_step = 5
learning_rate = 0.001
reward_decay = 0.98
e_greedy = 0.99
replace_target_iter = 1000
memory_size = 30000
batch_size = 128
e_greedy_increment = 1 / 10e4
output_graph = False

n_actions = 400     # 4*100, each action's value range is divided into 100 intervals
n_features = 10

env = DQNEnvTwoUsers(n_actions)

RL = DeepQNetwork(n_actions = n_actions,
                  n_features = n_features,
                  learning_rate = learning_rate,
                  reward_decay = reward_decay,
                  e_greedy = e_greedy,
                  replace_target_iter = replace_target_iter,
                  memory_size = memory_size,
                  batch_size = batch_size,
                  e_greedy_increment = e_greedy_increment,
                  output_graph = output_graph
                  )

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

num_data = 0

for ep in tqdm(range(EP_MAX), ncols=60):
    s, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
        Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2 = env.reset()
    ep_r = 0.
    all_r = []
    for t in range(EP_LEN):
        action = RL.choose_action(s)    # index

        if (EP_MAX - ep) < 5:
            a = action.copy()
            a[0] = (a[0] - n_actions/8) / (n_actions/8)       # index -> [-1, 1]
            a[1] = (a[1] - n_actions/4 - n_actions/8) / (n_actions/8)
            a[2] = (a[2] - n_actions/2 - n_actions/8) / (n_actions/8)
            a[3] = (a[3] - (n_actions/4)*3 - n_actions/8) / (n_actions/8)
            a[0] = a[0] * (power_max / 2) + (power_max / 2) + 1e-15       # [-1, 1] -> dBm
            a[1] = a[1] * (power_max / 2) + (power_max / 2) + 1e-15
            a[2] = a[2] * (rate_max / 2) + (rate_max / 2) + 1e-15
            a[3] = a[3] * (rate_max / 2) + (rate_max / 2) + 1e-15
            all_power_1.append(a[0]); all_power_2.append(a[1])
            all_rate_1.append(a[2]); all_rate_2.append(a[3])
            if t == 0:
                all_power_1_smooth.append(a[0]); all_power_2_smooth.append(a[1]) 
                all_rate_1_smooth.append(a[2]); all_rate_2_smooth.append(a[3])
            else:
                all_power_1_smooth.append(all_power_1_smooth[-1] * 0.9 + 0.1 * a[0])
                all_power_2_smooth.append(all_power_2_smooth[-1] * 0.9 + 0.1 * a[1])
                all_rate_1_smooth.append(all_rate_1_smooth[-1] * 0.9 + 0.1 * a[2])
                all_rate_2_smooth.append(all_rate_2_smooth[-1] * 0.9 + 0.1 * a[3])

        s_, reward, end, H_, \
        ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, \
        Q_len_1, Q_len_2, \
        sum_r_1, sum_p_1, sum_r_2, sum_p_2, energy_effi_1, energy_effi_2 = \
            env.step([action, H, ty1_sum_succ_1, ty1_sum_fail_1, ty1_sum_succ_2, ty1_sum_fail_2, 
                     Q_len_1, Q_len_2, sum_r_1, sum_p_1, sum_r_2, sum_p_2])  
        RL.store_transition(s, action, reward, s_)
        ep_r += reward
        all_r.append(reward)
        num_data += 1
        if num_data >= batch_size and num_data % not_learn_step == 0:
            RL.learn()
        s = s_
        H = H_

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
plt.suptitle('DQN training epoch = ' + str(EP_MAX) + ' training result')

path = './DQN_fig_result/'
if not os.path.exists(path):
    os.makedirs(path)

plt.savefig(path + 'DQN ' + str(EP_MAX) + ' train_result_' + 
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
plt.suptitle('DQN training epoch = ' + str(EP_MAX) + ' training result')
plt.savefig(path + 'DQN' + str(EP_MAX) + ' action_' + 
            datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'), dpi = 401)
plt.show()

