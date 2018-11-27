# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 05:31:14 2018

@author: tianye
"""

# 在 Python 中载入所需库
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


# 读入数据集，并对日访问量 y 取对数处理
df = pd.read_csv('./example_wp_log_peyton_manning.csv')
#df['y'] = np.log(df['y'])
df.head()

# 拟合模型
m = Prophet(changepoint_range=0.6, changepoint_prior_scale=1, n_changepoints=5)
m.fit(df);

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#
#趋势分解
#m.plot_components(forecast)

# You can plot the forecast by calling the Prophet.plot method and passing in your forecast dataframe.
fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
