train_file_name = r'..\dataset\adult_train.csv'
test_file_name = r'..\dataset\adult_test.csv'

category_columns = [1,3,5,6,7,8,9,13]

import pandas as pd
import numpy as np

train_data = pd.read_csv(train_file_name,header=None)
test_data = pd.read_csv(test_file_name,header=None)

# shuffle
np.random.seed(0) # set a random seed to replay
index = np.arange(train_data.shape[0])
np.random.shuffle(index)
train_data = train_data.iloc[index]
train_label =train_data.iloc[:,-1]
train_data =train_data.iloc[:,0:-1]
test_label = test_data.iloc[:,-1]
test_data =test_data.iloc[:,0:-1]

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=category_columns)
ohe.fit(pd.concat([train_data,test_data]))

# generate validate set
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,mean_squared_error

# normalize
# max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# train_data.apply(max_min_scaler)
# test_data.apply(max_min_scaler)

# cross validation
K = 0
for train_index,valid_index in kf.split(train_data):
    K = K + 1
    x_train,y_train = train_data.iloc[train_index],train_label.iloc[train_index]
    x_valid,y_valid = train_data.iloc[valid_index],train_label.iloc[valid_index]
    x_train = ohe.transform(x_train)
    x_valid = ohe.transform(x_valid)
    x_test = ohe.transform(test_data)
    lr = LogisticRegression()
    lr.fit(X=x_train,y=y_train)
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









