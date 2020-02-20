import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn
from dqn.CDQN import DeepQNetwork
from dqn.FeedbackMachine import DataLoad,Environment
import pandas as pd

def run(agent,env,episodes):
    if not isinstance(agent,DeepQNetwork) or not isinstance(env,Environment):
        raise KeyError('input error')
    step = 0
    for episode in range(episodes):

        observation = env.reset()

        while True:
            action = agent.choose_action(observation)
            # action = agent.choose_random_action(observation)
            # observation_,reward,done,_ = env.step_without_memory(action)
            observation_, reward, done, _ = env.step(action)
            agent.store_transition(observation,action,reward,observation_)

            if (step>200) and (step%5==0):
                print('After {} steps, learn!'.format(step))
                agent.learn()

            observation = observation_

            if done:
                print('for {} episode,break!'.format(episode))
                break
            step += 1
    print('run over!')

def eval(agent,env,category_columns):
    observation = env.reset()
    f_observation = env.reset()
    while True:
        action = agent.eval(observation)
        observation_,_,done,_= env.step(action)
        # observation_, _, done, _ = env.step_without_memory(action)
        observation = observation_

        if done:
            f_observation = observation
            break

    cols = []
    f_observation = list(f_observation)
    for i in range(len((f_observation))):
        if f_observation[i] == 1:
            cols.append(i)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,recall_score,mean_squared_error
    from sklearn.preprocessing import OneHotEncoder
    train_data, train_label, test_data, test_label = DataLoad(r'..\dataset\adult_train.csv',
                                                              r'..\dataset\adult_test.csv')
    train_data_ = train_data.iloc[:, cols]
    test_data_ = test_data.iloc[:, cols]

    intersection = []
    for i in range(len(cols)):
        if cols[i] in category_columns:
            intersection.append(i)

    ohe = OneHotEncoder(categorical_features=intersection)
    ohe.fit(pd.concat([train_data_, test_data_]))

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, random_state=0)

    # cross validation
    K = 0
    for train_index, valid_index in kf.split(train_data_):
        K = K + 1
        x_train, y_train = train_data_.iloc[train_index], train_label.iloc[train_index]
        x_valid, y_valid = train_data_.iloc[valid_index], train_label.iloc[valid_index]
        x_train = ohe.transform(x_train)
        x_valid = ohe.transform(x_valid)
        x_test = ohe.transform(test_data_)
        lr = LogisticRegression()
        lr.fit(X=x_train, y=y_train)
        train_pred = lr.predict(x_train)
        valid_pred = lr.predict(x_valid)
        test_pred = lr.predict(x_test)
        train_acc = accuracy_score(y_train, train_pred)
        valid_acc = accuracy_score(y_valid, valid_pred)
        test_acc = accuracy_score(test_label, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        valid_mse = mean_squared_error(y_valid, valid_pred)
        test_mse = mean_squared_error(test_label, test_pred)
        train_rec = recall_score(y_train, train_pred)
        valid_rec = recall_score(y_valid, valid_pred)
        test_rec = recall_score(test_label, test_pred)
        print("================================================================")
        print("{}th Cross Validation".format(K))
        print("train valid test accuracy : {} |{} |{}".format(train_acc, valid_acc, test_acc))
        print("train valid test mse      : {} |{} |{}".format(train_mse, valid_mse, test_mse))
        print("train valid test recall      : {} |{} |{}".format(train_rec, valid_rec, test_rec))

import time
if __name__=='__main__':
    env = Environment(14,r'..\dataset\adult_train.csv',r'..\dataset\adult_test.csv',[1, 3, 5, 6, 7, 8, 9, 13])
    agent = DeepQNetwork(env.n_action,env.total_feature_num,
                         learning_rate=0.01,
                         e_greedy=0.9,
                         replace_target_iter=100,
                         memory_size=2000)
    start = time.time()
    run(agent,env,2000)
    end = time.time()
    print('Time Cost:',end-start)
    agent.plot_cost(save=False)
    eval(agent, env, [1, 3, 5, 6, 7, 8, 9, 13])
