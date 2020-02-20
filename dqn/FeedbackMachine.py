import gym
import tensorflow as tf
import numpy as np
import pandas as pd
from gym.spaces import Discrete

class Environment(gym.Env):
    def __init__(self,total_feature_num,train_file,test_file,category_columns):
        self.total_feature_num = total_feature_num
        self.action_space = Discrete(total_feature_num)
        self.n_action = self.action_space.n
        self.state = [1]*total_feature_num
        self.train_data, self.train_label, self.test_data, self.test_label=DataLoad(train_file,test_file)
        self.category_columns = category_columns
        self.state_memory = []
        self.reward_memory = []
    def reset(self):
        # random start
        # init = []
        # for i in [0,0,1,1,1,1,1,1,0,1,1,1,1,0]:
        #     if i==0 and np.random.uniform() < 0.5:
        #         a = 1
        #     else:
        #         a = 1
        #     init.append(a)
        # fixed start
        init = [1] * self.total_feature_num
        self.state = np.array(init)
        return self.state

    def step(self,a):
        temp = []
        act = [0]*a + [1] + [0]*(self.total_feature_num-a-1)
        for i in range(len(act)):
            temp.append((act[i]+self.state[i])%2)
        self.state = np.array(temp)
        if temp not in self.state_memory:
            reward = get_reward(self.train_data,self.train_label,self.test_data,self.test_label,cols=self.state,category_columns=self.category_columns)
            reward = reward - 0.84*2
            self.state_memory.append(temp)
            self.reward_memory.append(reward)
        else:
            index = self.state_memory.index(temp)
            reward = self.reward_memory[index]
        print(self.state)
        if sum(self.state) == 11: # 10是被挑选的特征数量
            done = 1
        else:
            done = 0
        return self.state,reward,done,{}

    def step_without_memory(self,a):
        temp = []
        act = [0]*a + [1] + [0]*(self.total_feature_num-a-1)
        for i in range(len(act)):
            temp.append((act[i]+self.state[i])%2)
        self.state = np.array(temp)
        reward = get_reward(self.train_data,self.train_label,self.test_data,self.test_label,cols=self.state,category_columns=self.category_columns)
        reward = reward - 0.84*2
        print(self.state)
        if sum(self.state) == 11: # 10是被挑选的特征数量
            done = 1
        else:
            done = 0
        return self.state,reward,done,{}

class NeuralNetwork(object):
    def __init__(self,input_dim):
        self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.w = tf.Variable(tf.random_normal([input_dim, 1], stddev=1, dtype=tf.float32), name='weight')
        self.b = tf.Variable(tf.random_normal([1], stddev=1, dtype=tf.float32), name='bias')
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy())

    def forward(self):
        y_ = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b)
        return  y_

    def cross_entropy(self):
        y_ = self.forward()
        cross_entropy = -tf.reduce_mean(
            self.y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)) + (1 - self.y) * tf.log(tf.clip_by_value(1 - y_, 1e-10, 1.0)))
        return cross_entropy

def DataLoad(train_file,test_file,random_seed=False):
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)

    # shuffle
    if random_seed is not False:
        np.random.seed(random_seed)  # set a random seed to replay
    index = np.arange(train_data.shape[0])
    np.random.shuffle(index)
    train_data = train_data.iloc[index]
    train_label = train_data.iloc[:, -1]
    train_data = train_data.iloc[:, 0:-1]
    test_label = test_data.iloc[:, -1]
    test_data = test_data.iloc[:, 0:-1]

    return train_data,train_label,test_data,test_label

# train_data,train_label,test_data,test_label = DataLoad(r'..\dataset\adult_train.csv',r'..\dataset\adult_test.csv',[1, 3, 5, 6, 7, 8, 9, 13])

def get_reward(train_data,train_label,test_data,test_label,cols,category_columns):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder

    train_data_ = train_data.iloc[:, cols]
    test_data_ = test_data.iloc[:, cols]

    intersection = []
    for i in range(len(cols)):
        if cols[i] in category_columns:
            intersection.append(i)

    # print(intersection)
    if not len(intersection):
        ohe = OneHotEncoder(categorical_features=intersection)
        ohe.fit(pd.concat([train_data_, test_data_]))
        train_data_ = ohe.transform(train_data_)
        test_data_ = ohe.transform(test_data_)

    lr = LogisticRegression()
    lr.fit(X=train_data_, y=train_label)
    test_pred = lr.predict(test_data_)
    train_pred = lr.predict(train_data_)
    # accuracy_score(test_label, test_pred) +
    reward =  accuracy_score(train_label,train_pred)
    return reward

# train_data,train_label,test_data,test_label = DataLoad(r'..\dataset\adult_train.csv',r'..\dataset\adult_test.csv')
# reward = get_reward(train_data,train_label,test_data,test_label,cols=[1,2,3],category_columns=[1, 3, 5, 6, 7, 8, 9, 13])



