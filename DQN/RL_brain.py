"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

Based on Morvan's repo of TF1.x, rewrite with TF2.1
View more on Morvan's tutorial page: https://morvanzhou.github.io/tutorials/
"""

import datetime
import os
import numpy as np
import tensorflow as tf


# np.random.seed(1)
# tf.random.set_seed(1)

# Deep Q Network off-policy
class DeepQNetwork():
    def __init__(self, n_actions, n_features, learning_rate=0.01, 
        reward_decay=0.9, e_greedy_final=0.1, replace_target_iter=300, 
        memory_size=600, batch_size=32, e_greedy_decrement=0.001, 
        num_layers=2, num_units=10):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_final = e_greedy_final
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decrement = e_greedy_decrement
        self.epsilon = 1 if e_greedy_decrement is not None else self.epsilon_final
        self.num_layers = num_layers
        self.num_units = num_units

        self.print_eps = False
        # initialize evaluate network
        self.model = self.build_net()
        # target network
        self.target_model = self.build_net()
        self.tau = 0.1                          # soft update
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, not done, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.cost_his = []

    def build_net(self):
        model = tf.keras.Sequential()
        for _ in range(self.num_layers):
            model.add(tf.keras.layers.Dense(self.num_units, activation='relu'))
        model.add(tf.keras.layers.Dense(self.n_actions))
        return model

    def store_transition(self, s, a, r, done, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, not done, s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation.astype(np.float32)
        if len(observation.shape) < 2:
            observation = observation[np.newaxis, :]
        # forward feed the observation and get q value for every actions
        actions_value = self.model(observation)
        # random choice with probability epsilon
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(actions_value)
        return action, actions_value

    def choose_action_test(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation.astype(np.float32)
        if len(observation.shape) < 2:
            observation = observation[np.newaxis, :]
        # forward feed the observation and get q value for every actions
        actions_value = self.model(observation)
        action = np.argmax(actions_value)
        return action, actions_value

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            if self.learn_step_counter == 0:
                # initialize network
                _ = self.model(np.ones(shape=(1, self.n_features), dtype=np.float32))
                _ = self.target_model(np.ones(shape=(1, self.n_features), dtype=np.float32))
            self.target_model.set_weights(self.model.get_weights())
            # soft update
            # weights = self.model.get_weights()
            # target_weights = self.target_model.get_weights()
            # for i in range(len(target_weights)):  # set tau% of target model to be new weights
            #     target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            # self.target_model.set_weights(target_weights)
            # print('\ntarget_params_replaced\n')
            if self.print_eps:
                print('epsilon = {:.4f}'.format(self.epsilon))
                if self.epsilon <= self.epsilon_final:
                    self.print_eps = False
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :].astype(np.float32)
        # train evaluate network
        with tf.GradientTape() as tape:
            tape.watch(self.model.variables)
            q_eval = self.model(batch_memory[:, :self.n_features])
            q_next = self.target_model(batch_memory[:, -self.n_features:]).numpy()
            # change q_target w.r.t q_eval's action
            q_target = q_eval.numpy().copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(np.int32)
            reward = batch_memory[:, self.n_features + 1]
            q_target[batch_index, eval_act_index] = reward \
                + self.gamma * batch_memory[:, -self.n_features-1] * np.max(q_next, axis=1)
            loss = tf.keras.losses.mean_squared_error(y_true=q_target, y_pred=q_eval)
        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))
        self.cost_his.append(tf.reduce_sum(loss))
        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_final else self.epsilon_final
        self.learn_step_counter += 1

    def save_fn(self, path):
        self.model.save_weights(path)
