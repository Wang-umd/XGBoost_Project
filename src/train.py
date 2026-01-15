# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from log import Logger
from common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


def ana_data(data):
    data = data.copy(deep=True)
    # 1.数据整体情况
    print(data.info())
    print(data.head())
    fig = plt.figure(figsize=(20, 32))
    # 2.负荷整体的分布情况
    ax1 = fig.add_subplot(411)
    ax1.hist(data['power_load'], bins=100)
    ax1.set_title('负荷分布直方图')
    # 3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    ax2 = fig.add_subplot(412)
    data['hour'] = data['time'].str[11:13]
    data_hour_avg = data.groupby(by='hour', as_index=False)['power_load'].mean()
    ax2.plot(data_hour_avg['hour'], data_hour_avg['power_load'], color='b', linewidth=2)
    ax2.set_title('各小时的平均负荷趋势图')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('负荷')
    # 4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    ax3 = fig.add_subplot(413)
    data['month'] = data['time'].str[5:7]
    data_month_avg = data.groupby('month', as_index=False)['power_load'].mean()
    ax3.plot(data_month_avg['month'], data_month_avg['power_load'], color='r', linewidth=2)
    ax3.set_title('各月份平均负荷')
    ax3.set_xlabel('月份')
    ax3.set_ylabel('平均负荷')
    # 5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    ax4 = fig.add_subplot(414)
    data['week_day'] = data['time'].apply(lambda x: pd.to_datetime(x).weekday())
    data['is_workday'] = data['week_day'].apply(lambda x: 1 if x <= 4 else 0)
    power_load_workday_avg = data[data['is_workday'] == 1]['power_load'].mean()
    power_load_holiday_avg = data[data['is_workday'] == 0]['power_load'].mean()
    ax4.bar(x=['工作日平均负荷', '周末平均负荷'], height=[power_load_workday_avg, power_load_holiday_avg])
    ax4.set_ylabel('平均负荷')
    ax4.set_title('工作日与周末的平均负荷对比')
    plt.savefig('../data/fig/负荷分析图.png')


def feature_engineering(data, logger):
    logger.info("===============开始进行特征工程处理===============")
    result = data.copy(deep=True)
    logger.info("===============开始提取时间特征===================")
    result['hour'] = result['time'].str[11:13]
    result['month'] = result['time'].str[5:7]
    time_enconding = pd.get_dummies(result[['hour', 'month']])
    result = pd.concat([result, time_enconding], axis=1)
    logger.info("==============开始提取相近时间窗口中的负荷特征====================")
    load_1h = result['power_load'].shift(1)
    load_2h = result['power_load'].shift(2)
    load_3h = result['power_load'].shift(3)
    load_time = pd.concat([load_1h, load_2h, load_3h], axis=1)
    load_time.columns = ['前1小时', '前2小时', '前3小时']
    result = pd.concat([result, load_time], axis=1)
    logger.info("============开始提取昨日同时刻负荷特征===========================")
    result['yesterday_time'] = result['time'].apply(
        lambda x: (pd.to_datetime(x) - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    time_dict = result.set_index('time')['power_load'].to_dict()
    result['yesterday_load'] = result['yesterday_time'].apply(lambda x: time_dict.get(x))
    logger.info("============开始去掉空值整理并返回===========================")
    result = result.dropna()
    result_feature = list(time_enconding.columns) + list(load_time.columns) + ['yesterday_load']
    return result, result_feature


def model_train(data, features, logger):
    logger.info("=========开始模型训练===================")
    x = data[features]
    y = data['power_load']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=22)
    # # 2.网格化搜索与交叉验证
    # print("开始网格化搜索")
    # param_dict = {
    #     'n_estimators': [50, 100, 150, 200],
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.1, 0.01]
    # }
    # grid_cv = GridSearchCV(estimator=XGBRegressor(),
    #                        param_grid=param_dict, cv=5)
    # grid_cv.fit(x_train, y_train)
    # print(grid_cv.best_params_)  # {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 150}
    # 3.模型训练
    xgb = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1)
    xgb.fit(x_train, y_train)
    # 4.模型评价
    y_pred_train = xgb.predict(x_train)
    y_pred_test = xgb.predict(x_test)
    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_pred_train)
    print(f"模型在训练集上的均方误差：{mse_train}")
    print(f"模型在训练集上的平均绝对误差：{mae_train}")
    mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)
    print(f"模型在测试集上的均方误差：{mse_test}")
    print(f"模型在测试集上的平均绝对误差：{mae_test}")
    logger.info("=========================模型训练完成=============================")
    # 5.模型保存
    joblib.dump(xgb, '../model/xgb2.pkl')
    logger.info("=========================模型保存完成=============================")


class PowerLoadModel(object):
    def __init__(self, filename):
        logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        self.data_source = data_preprocessing('../data/train.csv')


if __name__ == '__main__':
    # 1.加载数据集
    model = PowerLoadModel('../data/train.csv')
    # 2.分析数据
    # ana_data(model.data_source)
    # 3.特征工程
    data, data_feature = feature_engineering(model.data_source, model.logfile)
    # 4.模型训练、模型评价与模型保存
    model_train(data, data_feature, model.logfile)
