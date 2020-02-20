import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn
import numpy as np
import pandas as pd


def DataLoad(train_file_name, test_file_name):
    train_data = pd.read_csv(train_file_name, header=None)
    test_data = pd.read_csv(test_file_name, header=None)
    return train_data, test_data


# shuffle
def DataShuffle(train_data, test_data):
    np.random.seed(0)  # set a random seed to replay
    index = np.arange(train_data.shape[0])
    np.random.shuffle(index)
    train_data = train_data.iloc[index]
    train_label = train_data.iloc[:, -1]
    train_data = train_data.iloc[:, 0:-1]
    test_label = test_data.iloc[:, -1]
    test_data = test_data.iloc[:, 0:-1]
    return train_data, train_label, test_data, test_label


# cols = train_data.columns.values.tolist()

from sklearn.ensemble import GradientBoostingClassifier


def GBDT_feature_selection_task(train_data, train_label, cols):
    gbc = GradientBoostingClassifier()
    gbc.fit(X=train_data, y=train_label)
    fi = list(gbc.feature_importances_)

    index = []
    for i in fi:
        if i > 0:
            index.append(fi.index(i))

    selected_index = []
    selected_score = []
    for i in index:
        selected_index.append(cols[i])
        selected_score.append(fi[i])
    return selected_index, selected_score


# _, gbdt_scores = GBDT_feature_selection_task(train_data, train_label, cols)

"""[0.06227524514929372, 0.004236916380586195, 0.0030101443467579784, 0.0001772627534679659, 0.204569398815289, 
0.023956693003479926, 0.022303245172975907, 0.3500663355924167, 0.0006222801334802581, 0.005629177550009482, 
0.2222695533043123, 0.061563304943371165, 0.038241340317296244, 0.001079102537263184]"""


def TopK(gbdt_scores, K):
    import heapq
    selected_index = list(map(gbdt_scores.index, heapq.nlargest(K, gbdt_scores)))
    selected_score = heapq.nlargest(K, gbdt_scores)
    print(selected_score)
    print(selected_index)
    return selected_score, selected_index


"""top 10:"""
"""[7, 10, 4, 0, 11, 12, 5, 6, 9, 1]"""


def SelectData(train_data, test_data, selected_cols):
    # selected_cols = [7, 10, 4, 0, 11, 12, 5, 6, 9, 1]
    train_data = train_data[selected_cols]
    test_data = test_data[selected_cols]
    return train_data, test_data


# category_column = [1, 5, 6, 7, 9]

# normalize
# max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# train_data.apply(max_min_scaler)
# test_data.apply(max_min_scaler)
def OneHot(train_data, test_data, category_columns):
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(categorical_features=category_columns)
    ohe.fit(pd.concat([train_data, test_data]))
    return ohe


# cross validation
def CrossValidation(train_data, train_label, test_data, test_label, ohe=None, one_hot=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,mean_squared_error,recall_score
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, random_state=0)
    K = 0
    for train_index, valid_index in kf.split(train_data):
        K = K + 1
        x_train, y_train = train_data.iloc[train_index], train_label.iloc[train_index]
        x_valid, y_valid = train_data.iloc[valid_index], train_label.iloc[valid_index]
        if one_hot == True:
            x_train = ohe.transform(x_train)
            x_valid = ohe.transform(x_valid)
            x_test = ohe.transform(test_data)
        else:
            x_test = test_data
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
        print("train valid test accuracy : {} |{} |{}".format(train_acc,valid_acc,test_acc))
        print("train valid test mse      : {} |{} |{}".format(train_mse, valid_mse, test_mse))
        print("train valid test recall      : {} |{} |{}".format(train_rec, valid_rec, test_rec))


if __name__ == '__main__':
    train_file_name = r'..\dataset\adult_train.csv'
    test_file_name = r'..\dataset\adult_test.csv'
    category_columns = [1, 3, 5, 6, 7, 8, 9, 13]
    train_data, test_data = DataLoad(train_file_name, test_file_name)
    train_data, train_label, test_data, test_label = DataShuffle(train_data, test_data)
    cols = train_data.columns.values.tolist()
    _, gbdt_scores = GBDT_feature_selection_task(train_data, train_label, cols)
    selected_scores, selected_index = TopK(gbdt_scores, 11)
    temp = []
    for i in selected_index:
        if i in category_columns:
            temp.append(i)

    train_data, test_data = SelectData(train_data, test_data, selected_index)
    ohe = OneHot(train_data, test_data, category_columns=temp)
    CrossValidation(train_data, train_label, test_data, test_label,ohe,one_hot=True)
