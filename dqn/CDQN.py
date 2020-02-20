import numpy as np
import tensorflow as tf
import random
np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate = 0.01, # DQN的参数更新学习率
            reward_decay = 0.9, # 回报折扣率
            e_greedy = 0.9, # e-贪心探索的概率
            memory_size = 500, # Replay Buffer的容量
            replace_target_iter = 300,
            batch_size = 32,
            e_greedy_increment = None,
            output_graph = False,
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

        # 总共的学习步数
        self.learn_step_counter = 0

        # 初始化记录，n_features*2+2是因为[s,a,r,s_]中a和r各占1
        self.memory = np.zeros((self.memory_size,n_features*2+2))

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # -------------------------------------all inputs------------------------------------------
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        self.r = tf.placeholder(tf.float32,[None,],name='r')
        self.a = tf.placeholder(tf.int32,[None,],name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # -------------------------------- build evaluate_net -------------------------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(inputs=self.s,units=20,activation=tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            self.q_eval = tf.layers.dense(e1,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='q')

         # -------------------------------- build target_net -------------------------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t1')
            self.q_next = tf.layers.dense(t1,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r +self.gamma * tf.reduce_max(self.q_next,axis=1,name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0],dtype=tf.int32),self.a],axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval,indices=a_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_wrt_a,name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s,[a,r],s_))
        # 用新的memory替换掉旧的memory
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self,observation):
        # 增加batch这一维度以便feed到placeholder里去
        observation_ = observation[np.newaxis,:]
        if  np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval,feed_dict={self.s:observation_})
            action_value = action_value - np.min(action_value)
            action = np.argmax(action_value*observation)
        else:
            temp = []
            for i in range(len(observation)):
                if observation[i] == 1:
                    temp.append(i)
            if temp != []:
                action = random.sample(temp,1)[0]
            else:
                action = np.random.randint(0,self.n_actions)

        return action

    def choose_random_action(self,observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def eval(self,observation):
        observation_ = observation[np.newaxis, :]
        action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation_})
        action_value = action_value - np.min(action_value)
        action = np.argmax(action_value * observation)
        return action

    def learn(self):
        # 判断是否到了需要更新target网络的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # 从memory中采样出一个batch的memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)

        batch_memory = self.memory[sample_index,:]

        _,cost = self.sess.run([self._train_op,self.loss],
                               feed_dict={
                                   self.s:batch_memory[:,:self.n_features],
                                   self.a:batch_memory[:,self.n_features],
                                   self.r:batch_memory[:,self.n_features+1],
                                   self.s_:batch_memory[:,-self.n_features:],
                               })

        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self,save=True):
        import matplotlib.pyplot as plt
        if save == True:
            with open("choose_random_action_10.txt","w") as f:
                for i in self.cost_his:
                    f.write(str(i)+'\t')
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


