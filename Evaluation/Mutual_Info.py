from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from Evaluation.GBDT import DataLoad,DataShuffle,CrossValidation,SelectData,OneHot

train_file_name = r'..\dataset\adult_train.csv'
test_file_name = r'..\dataset\adult_test.csv'

category_columns = [1, 3, 5, 6, 7, 8, 9, 13]
train_data, test_data = DataLoad(train_file_name, test_file_name)
train_data, train_label, test_data, test_label = DataShuffle(train_data, test_data)
cols = train_data.columns.values.tolist()

skb = SelectPercentile(mutual_info_classif,percentile=78)
skb.fit(train_data,train_label)
result = list(skb.get_support())
selected_index = []

for i in range(len(result)):
    if result[i]:
        selected_index.append(i)
temp = []
for i in selected_index:
    if i in category_columns:
        temp.append(i)

train_data, test_data = SelectData(train_data, test_data, selected_index)
ohe = OneHot(train_data, test_data, category_columns=temp)
CrossValidation(train_data, train_label, test_data, test_label,ohe,one_hot=True)
# skb_filter = train_data.columns[skb.get_support()]
# print(skb_filter)