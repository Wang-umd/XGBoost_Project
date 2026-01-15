import pandas as pd
import numpy as np


def data_preprocessing(fp):

    # 1.获取数据源
    data = pd.read_csv(fp)
    # 2.时间格式化
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # 3.按时间升序排序
    data.sort_values(by='time', ascending=True, inplace=True)
    # 4.去重
    data.drop_duplicates(inplace=True)
    return data


def mean_absolute_percentage_error(y_true, y_pred):

    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred have different number of output ")
    abs_percentage_error = np.abs((y_true - y_pred) / y_true)
    return np.sum(abs_percentage_error) / n * 100