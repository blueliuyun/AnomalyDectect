# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:13:53 2019

@author: tianye
"""
import matplotlib.pyplot as plt

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
    
download_robot_execution_failures()

timeseries, y = load_robot_execution_failures()

print(timeseries.head())

#
timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True, figsize=(10,10))
plt.show()

#
timeseries[timeseries['id'] == 21].plot(subplots=True, sharex=True, figsize=(10,10))
plt.show()