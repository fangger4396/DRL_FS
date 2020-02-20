from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif
from Evaluation.GBDT import DataLoad,DataShuffle,CrossValidation,SelectData,OneHot
from sklearn.linear_model import LogisticRegression
train_file_name = r'..\dataset\adult_train.csv'
test_file_name = r'..\dataset\adult_test.csv'

category_columns = [1, 3, 5, 6, 7, 8, 9, 13]
train_data, test_data = DataLoad(train_file_name, test_file_name)
train_data, train_label, test_data, test_label = DataShuffle(train_data, test_data)
cols = train_data.columns.values.tolist()

estimator = LogisticRegression()
rfe = RFE(estimator=estimator, n_features_to_select=11)
rfe.fit(train_data,train_label)

result = list(rfe.support_)
selected_index = []

for i in range(len(result)):
    if result[i]:
        selected_index.append(i)
print(selected_index)
temp = []
for i in range(len(selected_index)):
    if selected_index[i] in category_columns:
        temp.append(i)
train_data, test_data = SelectData(train_data, test_data, selected_index)
ohe = OneHot(train_data, test_data, category_columns=temp)
CrossValidation(train_data, train_label, test_data, test_label,ohe,one_hot=True)