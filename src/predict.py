# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import datetime
from log import Logger
from common import data_preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


def pred_feature_extract(data_dict, time, logger):
    logger.info(f'=========解析预测时间为：{time}所对应的特征==============')
    # 特征列清单
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     '前1小时', '前2小时', '前3小时', 'yesterday_load']
    # 小时特征数据，使用列表保存起来
    hour_part = []
    pred_hour = time[11:13]
    for i in range(24):
        if pred_hour == feature_names[i][5:7]:
            hour_part.append(1)
        else:
            hour_part.append(0)
    # 月份特征数据，使用列表保存起来
    month_part = []
    pred_month = time[5:7]
    for i in range(24, 36):
        if pred_month == feature_names[i][6:8]:
            month_part.append(1)
        else:
            month_part.append(0)
    # 前1小时负荷
    last_1h_time = (pd.to_datetime(time) - pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time, 600)
    # 前2小时负荷
    last_2h_time = (pd.to_datetime(time) - pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 600)
    # 前3小时负荷
    last_3h_time = (pd.to_datetime(time) - pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 600)

    # 昨日同时刻负荷
    last_day_time = (pd.to_datetime(time) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S')
    last_day_load = data_dict.get(last_day_time, 600)

    # 特征数据，包含小时特征数据，月份特征数据，历史负荷数据
    feature_list = hour_part + month_part + [last_1h_load, last_2h_load, last_3h_load, last_day_load]
    feature_df = pd.DataFrame([feature_list], columns=feature_names)
    return feature_df


def prediction_plot(data):
    # 绘制在新数据下
    fig = plt.figure(figsize=(40, 20))
    ax = fig.add_subplot()
    # 绘制时间与真实负荷的折线图
    ax.plot(data['预测时间'], data['真实负荷'], label='真实负荷')
    # 绘制时间与预测负荷的折线图
    ax.plot(data['预测时间'], data['预测负荷'], label='预测负荷')
    ax.set_ylabel('负荷')
    ax.set_title('预测负荷以及真实负荷的折线图')
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    ax.xaxis.set_major_locator(mick.MultipleLocator(50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('../data/fig/预测效果.png')


class PowerLoadPredict(object):
    def __init__(self, filename):
        logfile_name = "predict_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        self.data_source = data_preprocessing(filename)
        self.data_dict = self.data_source.set_index('time')['power_load'].to_dict()


if __name__ == '__main__':
    # 2.定义电力负荷预测类(PowerLoadPredict)，配置日志，获取数据源、历史数据转为字典（避免频繁操作dataframe，提高效率）
    pred_obj = PowerLoadPredict('../data/test.csv')
    # 3.加载模型
    estimater = joblib.load('../model/xgb2.pkl')
    # 4.模型预测
    pred_times = pred_obj.data_source[pred_obj.data_source['time'] >= '2015-08-01 00:00:00']['time']

    evaluate_list = []
    for pred_time in pred_times:
        # print(f"开始预测时间为：{pred_time}的负荷")
        pred_obj.logfile.info(f"开始预测时间为：{pred_time}的负荷")
        data_his_dict = {k: v for k, v in pred_obj.data_dict.items() if k < pred_time}
        # 4.3预测负荷
        processed_data = pred_feature_extract(data_his_dict, pred_time, pred_obj.logfile)
        pred_value = estimater.predict(processed_data)
        true_value = pred_obj.data_dict.get(pred_time, 500)
        pred_obj.logfile.info(f"开始预测时间为{pred_time}的负荷,真实值为{true_value},预测值为{pred_value[0]}")
        evaluate_list.append([pred_time, true_value, pred_value[0]])

    evaluate_df = pd.DataFrame(evaluate_list, columns=['预测时间', '真实负荷', '预测负荷'])
    print(evaluate_df)
    prediction_plot(evaluate_df)
