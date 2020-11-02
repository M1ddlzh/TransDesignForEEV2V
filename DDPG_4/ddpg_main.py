import os
import gym
import torch
import argparse
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DDPGPolicy
from tianshou.data import ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from my_offpolicy import offpolicy_trainer
from my_ddpg_collector import Collector

from env_fourusers_ddpg import EnvFourUsers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fourusers')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--step_per_epoch', type=int, default=30)
    parser.add_argument('--collect_per_step', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--layer_num', type=int, default=4)
    parser.add_argument('--unit_num', type=int, default=128)
    parser.add_argument('--training_num', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='mylog')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def training_ddpg(args=get_args()):
    env = EnvFourUsers(args.step_per_epoch)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    train_envs = VectorEnv(
        [lambda: EnvFourUsers(args.step_per_epoch) for _ in range(args.training_num)])
    test_envs = VectorEnv(
        [lambda: EnvFourUsers(args.step_per_epoch) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, 0, device=args.device,
        hidden_layer_size=args.unit_num)
    actor = Actor(net, args.action_shape, args.max_action,
        args.device, hidden_layer_size=args.unit_num).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # print(net, actor.last)
    net = Net(args.layer_num, args.state_shape,
        args.action_shape, concat=True, device=args.device, 
        hidden_layer_size=args.unit_num)
    critic = Critic(net, args.device, args.unit_num).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    # print(net, critic.last)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        args.tau, args.gamma, OUNoise(sigma=args.exploration_noise),
        # GaussianNoise(sigma=args.exploration_noise),
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=True, ignore_done=True)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ddpg')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # writer = SummaryWriter(log_path)
    writer = None
    # policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
    # print('relode model!')

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= 1e16

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn,
        writer=writer)
    train_collector.close()
    test_collector.close()

def testing_ddpg(args=get_args()):
    env = EnvFourUsers(args.step_per_epoch)
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    # model
    net = Net(args.layer_num, args.state_shape, 0, device=args.device,
        hidden_layer_size=args.unit_num)
    actor = Actor(net, args.action_shape, args.max_action,
        args.device, hidden_layer_size=args.unit_num).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(args.layer_num, args.state_shape,
        args.action_shape, concat=True, device=args.device, 
        hidden_layer_size=args.unit_num)
    critic = Critic(net, args.device, args.unit_num).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        args.tau, args.gamma, OUNoise(sigma=args.exploration_noise),
        # GaussianNoise(sigma=args.exploration_noise),
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=True, ignore_done=True)
    # restore model
    log_path = os.path.join(args.logdir, args.task, 'ddpg')
    policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
    print('\nrelode model!')

    env = EnvFourUsers(args.step_per_epoch)
    collector = Collector(policy, env)
    ep = 10000
    result = collector.collect(n_episode=ep, render=args.render)
    print('''\nty1_succ_1: {:.2f}, q_len_1: {:.2f}, 
        \nty1_succ_2: {:.2f}, q_len_2: {:.2f}, 
        \nty1_succ_3: {:.2f}, q_len_3: {:.2f}, 
        \nty1_succ_4: {:.2f}, q_len_4: {:.2f}, 
        \nee_1: {:.2f}, ee_2: {:.2f}, ee_3: {:.2f}, ee_4: {:.2f}, 
        \navg_rate:{:.2f}, \navg_power:{:.2f}\n'''
        .format(
        result["ty1s_1"][0]/ep, result["ql_1"][0]/ep, 
        result["ty1s_2"][0]/ep, result["ql_2"][0]/ep,
        result["ty1s_3"][0]/ep, result["ql_3"][0]/ep, 
        result["ty1s_4"][0]/ep, result["ql_4"][0]/ep, 
        result["ee_1"][0]/ep, result["ee_2"][0]/ep, 
        result["ee_3"][0]/ep, result["ee_4"][0]/ep, 
        result["avg_r"]/ep, result["avg_p"]/ep))
    print('large than Qmax: users1: {}, users2: {}, users3: {}, users4: {}.'
        .format(str(env.large_than_Q_1), str(env.large_than_Q_2), 
        str(env.large_than_Q_3), str(env.large_than_Q_4)))
    collector.close()


if __name__ == '__main__':
    start_time = datetime.now()
    training_ddpg()
    testing_ddpg()
    end_time = datetime.now()
    print(datetime.now())
    print('total time:', (end_time - start_time).total_seconds() / 60, 'minutes.')
