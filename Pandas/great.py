import pandas as pd
from datetime import datetime
import numpy as np
from pylab import *

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
rcParams['figure.figsize'] = 15,5
# figsize(15, 5)
# , index_col='Date' parse_dates=['Date'], index_col='Date'
df_train = pd.read_csv('./train.csv', parse_dates=['Date'], index_col='Date')
# df_test = pd.read_csv('./mock.test', parse_dates=['Date'])
# df_train_train.index.name = 'Date'

# start = datetime(2010, 2, 5)
# train_end = datetime(2012, 10, 26)
# test_start = datetime(2012, 11, 2)
# test_end = datetime(2013, 7, 26)

# df_dates_all = pd.date_range(start, test_end, freq="W-FRI")
# df_dates_all = pd.DataFrame(pd.Series(df_dates_all), columns=['Date'])

# merged = pd.merge(df_dates_all, df_train, on='Date', how='left', suffixes=('_all', '_train'))
# merged = merged.ffill()


# merged = pd.merge(merged, df_test, on='Date', how='right', suffixes=('_all', '_test'))


# print merged
where_train_store = df_train['Store'] == 1
where_train_dept = df_train['Dept'] == 85
df_got = df_train[where_train_store & where_train_dept]
subplot(111, axisbg='#ffffff')
df_got['Weekly_Sales'].plot(color='#000000', linewidth='1.75')


