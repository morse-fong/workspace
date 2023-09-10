#%%
# 导入所需的包
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 读取数据
attachment1 = pd.read_excel("CUMCM2023Problems/C题/6 个蔬菜品类的商品信息.xlsx")
attachment2 = pd.read_excel("CUMCM2023Problems/C题/销售流水明细数据.xlsx")
attachment3 = pd.read_excel("CUMCM2023Problems/C题/蔬菜类商品的批发价格.xlsx")


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Group by the date and product ID to get daily sales volume for each product
# Merge with attachment1 to get the category information/
daily_sales = attachment2.merge(attachment1, on='单品编码', how='left')

# Grouping the data by category and date
category_sales2 = daily_sales.groupby(['销售日期', '分类名称'])['销量(千克)'].mean().unstack()
category_sales2.replace(np.nan, 0, inplace=True)

def Lstm(category_sales):
    look_back = 7
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(category_sales.values.reshape(-1, 1))

    # 将时间序列数据转换为LSTM所需的格式
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    X, y = create_dataset(scaled_data, look_back)

    # # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(category_sales.values.reshape(-1, 1))

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    X, y = create_dataset(scaled_data, look_back)
    # 将数据划分为训练集和测试集
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=25, verbose=2)



    def create_dataset0(dataset, look_back=1):
        dataX = []
        for i in range(len(dataset) - look_back * 2 - 1, len(dataset) - look_back - 1):
            a = [dataset[i:(i + look_back), 0]]
            dataX.append(a)
        return np.array(dataX)

    unx = create_dataset0(scaled_data, look_back)
    predict = model.predict(unx)
    predict = scaler.inverse_transform(predict)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 反归一化
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])

    # 计算root mean squared error
    train_score = np.sqrt(np.mean((train_predict - y_train) ** 2))
    print('Train Score: %.2f RMSE' % (train_score))
    test_score = np.sqrt(np.mean((test_predict - y_test) ** 2))
    print('Test Score: %.2f RMSE' % (test_score))

    X_train

    plt.plot(scaler.inverse_transform(scaled_data))
    train_predict_plot = np.zeros_like(scaled_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.zeros_like(scaled_data)

    # print(scaler.inverse_transform(scaled_data))
    test_predict_plot[len(train_predict) + look_back:len(scaled_data) - 1, :] = test_predict

    predict_plot = np.zeros_like(scaled_data)
    predict_plot[len(predict_plot)-look_back:, :] = predict
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.plot(predict_plot)

    plt.show()

    print(predict)
for i in category_sales2.columns.values:
    category_sales0 = category_sales2[i]
    Lstm(category_sales0)
# '水生根茎类' '花叶类' '花菜类' '茄类' '辣椒类' '食用菌'

