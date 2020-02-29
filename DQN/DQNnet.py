import os
import numpy as np
import tensorflow as tf


np.random.seed(0)
tf.set_random_seed(0)

class DeepQNetwork:
    """
    DQN network
    """
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
            e_greedy_increment,
            output_graph
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.output_graph = output_graph

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 1 + 4))

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        self.q_eval, pi_params = self._build_eval_net('eval_net', trainable=True)
        self.q_next, oldpi_params = self._build_target_net('target_net', trainable=False)
        self.replace_target_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        self.sess = tf.Session()

        if self.output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_eval_net(self, name, trainable):
        """
        network for generating the value of all actions in the current state
        """
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu, trainable=trainable)
            for _ in range(3):
                l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            action_val = tf.layers.dense(l1, self.n_actions, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return action_val, params

    def _build_target_net(self, name, trainable):
        """
        network for generating the value of all actions in the next state
        """
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s_, 100, tf.nn.relu, trainable=trainable)
            for _ in range(3):
                l1 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            action_val = tf.layers.dense(l1, self.n_actions, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return action_val, params

    def store_transition(self, s, a, r, s_):
        """
        store training trajectory
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        # np.random.shuffle(self.memory)
        self.memory_counter += 1

    def choose_action(self, observation):
        """
        choose an action based on epsilon-greedy policy
        """
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        rand = np.random.uniform()
        if rand < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}).squeeze()
            p1, p2 = np.argmax(actions_value[0 : int(self.n_actions/4)]), \
                     np.argmax(actions_value[int(self.n_actions/4) : int(self.n_actions/2)])
            r1, r2 = np.argmax(actions_value[int(self.n_actions/2) : int(self.n_actions/4)*3]), \
                     np.argmax(actions_value[int(self.n_actions/4)*3 : int(self.n_actions)])
            action = [p1, p2+int(self.n_actions/4), r1+int(self.n_actions/2), r2+int(self.n_actions/4)*3]
        else:
            p1, p2 = np.random.randint(0, int(self.n_actions/4)), \
                     np.random.randint(int(self.n_actions/4), int(self.n_actions/2))
            r1, r2 = np.random.randint(int(self.n_actions/2), int(self.n_actions/4)*3), \
                     np.random.randint(int(self.n_actions/4)*3, int(self.n_actions))
            action = [p1, p2, r1, r2]

        return action

    def learn(self):
        """
        training the DQN agent
        """
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features : self.n_features+4].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index[:, 0]] = reward + \
            self.gamma * np.max(q_next[:, 0 : int(self.n_actions/4)], axis=1)
        q_target[batch_index, eval_act_index[:, 1]] = reward + \
            self.gamma * np.max(q_next[:, int(self.n_actions/4) : int(self.n_actions/2)], axis=1)
        q_target[batch_index, eval_act_index[:, 2]] = reward + \
            self.gamma * np.max(q_next[:, int(self.n_actions/2) : int(self.n_actions/4*3)], axis=1)
        q_target[batch_index, eval_act_index[:, 3]] = reward + \
            self.gamma * np.max(q_next[:, int(self.n_actions/4*3) : int(self.n_actions)], axis=1)

        # train eval network
        _, _ = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_model(self):
        """
        save graph and weights
        """
        self.saver.save(self.sess, './DQN_model')
