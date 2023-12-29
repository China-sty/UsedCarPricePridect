import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

train_path = os.path.join(os.path.dirname(__file__), '../data/used_car_train_20200313.csv')
test_path = os.path.join(os.path.dirname(__file__), '../data/used_car_testB_20200421.csv')

train_data = pd.read_csv(train_path, sep=' ')
test_data = pd.read_csv(test_path, sep=' ')
concat_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
# print(concat_data.head())
columns = concat_data.columns

concat_data['regDate'] = pd.to_datetime(concat_data['regDate'], format='%Y%m%d', errors='coerce')
concat_data['creatDate'] = pd.to_datetime(concat_data['creatDate'], format='%Y%m%d', errors='coerce')
# print(concat_data[['regDate','creatDate']].head())
concat_data['used_time'] = (concat_data['creatDate'] - concat_data['regDate']).dt.days
# print(concat_data['used_time'].head())

concat_data['creatDate_year'] = concat_data['creatDate'].dt.year
concat_data['creatDate_month'] = concat_data['creatDate'].dt.month
concat_data['creatDate_day'] = concat_data['creatDate'].dt.day
concat_data['regDate_year'] = concat_data['regDate'].dt.year
concat_data['regDate_month'] = concat_data['regDate'].dt.month
concat_data['regDate_day'] = concat_data['regDate'].dt.day
concat_data.drop(['regDate', 'creatDate','SaleID','seller','offerType','name'], axis=1, inplace=True)
concat_data.replace('-', np.nan, inplace=True)

descrete_feature = []
for column in concat_data.columns:
    if column not in ['price']:
        if(concat_data[column].nunique()<100):
            descrete_feature.append(column)
            # print(concat_data[column].value_counts())
            # print("-----------------------------------")
# concat_data['notRepairedDamage'].fillna(0.5, inplace=True)
# print(concat_data['notRepairedDamage'].value_counts())
# fig = plt.figure(figsize=(4,6))
# sns.boxplot(concat_data['notRepairedDamage'],orient = "V",width = 0.5)
# plt.show()
print(descrete_feature)
print(concat_data.head())
concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].apply(lambda x: float(x))

# for column in concat_data.columns:
#     if column not in descrete_feature:
#         figure = plt.figure(figsize=(4,6))
#         sns.histplot(concat_data[column], kde=True)
#         plt.title(column)
#         plt.show()

# concat_data['power'] = stats.boxcox(concat_data['power'])
# concat_data['v_4'] = stats.boxcox(concat_data['v_4'])

# 获取 DataFrame 的所有列名
search_columns = concat_data.columns.drop(descrete_feature).drop('price')

# 设置子图的行数和列数
num_rows = 4  # 行数
num_cols = 5  # 列数

# 设置子图的大小
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

# 遍历每个属性，绘制直方图
for i, column in enumerate(search_columns):
    ax = axes[i] if num_rows == 1 else axes[i // num_cols, i % num_cols]
    sns.histplot(data=concat_data, x=column, ax=ax)
    ax.set_title(column)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()


data_with_label = concat_data[concat_data['price'].notnull()]
# print(data_with_label.info())
# notRepairedDamage  125676
# regDate  138653
# bodyType           145494
# fuelType           141320
# gearbox            144019


data_no_label = concat_data[concat_data['price'].isnull()]
# print(data_with_label.shape)
# print(data_with_label.columns)

X = data_with_label.drop(['price'], axis=1)
y = data_with_label['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rf_model = LinearRegression()
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5)
# rf_model = LGBMRegressor(learning_rate = 0.01, n_estimators = 5000,num_leaves=1000, max_depth = 10, min_child_samples = 20, subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 0.005, n_jobs = -1)
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)
# mae = np.mean(abs(y_pred - y_test))
# print('rf_model score: ', rf_model.score(X_test, y_test))
# print('rf_model mae: ', mae)