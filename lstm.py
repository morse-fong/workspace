#%%
# 导入所需的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import zipfile

# 读取数据
attachment1 = pd.read_excel("CUMCM2023Problems/C题/6 个蔬菜品类的商品信息.xlsx")
attachment2 = pd.read_excel("CUMCM2023Problems/C题/销售流水明细数据.xlsx")
attachment3 = pd.read_excel("CUMCM2023Problems/C题/蔬菜类商品的批发价格.xlsx")
look_back = 3
daily_sales = attachment2.merge(attachment1, on='单品编码', how='left')


def Lstm(category_sales):
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(category_sales.values.reshape(-1, 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 将时间序列数据转换为LSTM所需的格式
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)


    X, y = create_dataset(scaled_data, look_back)


    #%%
    # 将数据划分为训练集和测试集
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    print(X_train.shape,X_test.shape)
    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #%%
    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=25, verbose=2)

    model.save('/model')

    # from keras.models import load_model
    # model = load_model('//model')

    # 预测
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    print(X_train.shape)
    print(train_predict.shape)
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


    #%%
    # 绘图
    plt.plot(scaler.inverse_transform(scaled_data))
    train_predict_plot = np.empty_like(scaled_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.empty_like(scaled_data)

    # print(scaler.inverse_transform(scaled_data))
    test_predict_plot[len(train_predict) + (look_back * 1):len(scaled_data) - 1, :] = test_predict
    # plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()
#%% md
# 2. 建立LSTM模型
# 接下来，您需要定义LSTM的结构并编译模型。
#%% md
# 3. 训练模型
# 使用您的训练数据训练LSTM模型。
#%% md
# 4. 进行预测
# 使用训练好的LSTM模型对测试数据进行预测。
#%% md
# 预测未来一周的销售量：
# 
# 使用LSTM或其他时间序列预测模型预测每个蔬菜品类的未来一周销售量。
# 确定补货策略：
# 
# 基于预测销售量，加上安全库存（以应对预测误差）来确定补货量。
# 考虑损耗率来进一步调整补货量。例如，如果某个蔬菜的损耗率为10%，那么您可能需要增加10%的补货量以应对这种损耗。
# 制定定价策略：
# 
# 使用“成本加成定价”方法：先确定每个蔬菜品类的成本，然后加上预期的利润率来确定售价。
# 考虑市场竞争、季节性因素和其他因素来调整价格。
# 如果蔬菜品相变差，可以提供折扣以加速销售。
# 也可以考虑使用动态定价策略，例如，如果某天销售量低于预期，可以稍微降低价格以吸引更多顾客。
# 最大化收益的策略：
# 
# 根据预测的销售量、定价策略和成本来模拟未来一周的预期收益。
# 使用优化算法（如线性规划）来确定能够最大化收益的最佳定价和补货策略。

# Grouping the data by category and date
category_sales2 = daily_sales.groupby(['销售日期', '分类名称'])['销量(千克)'].mean().unstack()
category_sales2.replace(np.nan, 0, inplace=True)
category_sales=category_sales2['水生根茎类']
Lstm(category_sales)