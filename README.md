# Deep Reinforcement Learning-aided Transmission Design for Multi-user V2V Networks
IEEE WCNC 2021.

This repository stores the source code of the paper "Deep Reinforcement Learning-aided Transmission Design for Multi-user V2V Networks".

## Requirements
- tensorflow==2.1.0
- pytorch==1.6.0
- tianshou==0.2.5
- gym==0.17.2
- cvxpy==1.0.25
- openpyxl

## Explanation
DDPG transmission desion algorithm is saved in folders with the name beginning with "ddpg_". Other benchmark methods are also stored in the corresponding folders.

DDPG is based on [Tianshou](https://github.com/thu-ml/tianshou) and is written with Pytorch1.6. DQN refers to [Morvan's model](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) and is rewritten with TF2.1.

OT, NOT and LYA algorithm rely on cvxpy. Windows 10: cvxpy 1.0.21 and Ubuntu 18.04: cvxpy 1.0.27 have been tested.
