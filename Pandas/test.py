import pandas as pd
from datetime import datetime
import numpy as np
# from pylab import *
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
# rcParams['figure.figsize'] = 15,5
# figsize(15, 5)
# , index_col='Date'
df_train_csv = pd.read_csv('./mock.train', parse_dates=['Date'])
# df_train_csv.index.name = 'Date'

start = datetime(2010, 2, 5)
train_end = datetime(2012, 10, 26)

df_train_fly = pd.date_range(start, train_end, freq="W-FRI")
df_train_fly = pd.DataFrame(pd.Series(df_train_fly), columns=['Date'])
df_train_fly["Store"] = 1

# merged = df_train_csv.join(df_train_fly.set_index(['Date']), on = ['Date'], how = 'inner')
# merged = pd.merge(df_train_csv, df_train_fly, on='Store', how='outer', lsuffix="_review")
# merged = df_train_fly.join(df_train_csv, on = ['Date'], how = 'left', lsuffix='_x')
# merged = df_train_fly.join(df_train_csv, how = 'left', lsuffix='_fly')
# merged = df_train_csv.merge(df_train_fly.set_index(['Date']), left_index=True, right_index=True, how = 'right')
# merged = df_train_csv.join(df_train_fly.set_index(['Date']), on = ['Date'], how = 'inner')
# merged = pd.merge(df_train_csv, df_train_fly, on='Store', how='outer', lsuffix="_review")
# merged = df_train_fly.join(df_train_csv, on = ['Date'], how = 'left', lsuffix='_x')
merged = df_train_fly.join(df_train_csv, how = 'left', lsuffix='_fly')
# merged = df_train_csv.merge(df_train_fly.set_index(['Date']), left_index=True, right_index=True, how = 'right')

merged = pd.merge(df_train_fly, df_train_csv, on='Date', how='left', suffixes=('_all', '_part'))
merged = merged.ffill()







# dfTrain['Weekly_Sales'].plot()