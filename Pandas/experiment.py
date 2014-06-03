import sys
from pprint import pprint
import numpy as np
import scipy
import math
from sklearn import linear_model, svm, tree, neighbors
from sklearn.naive_bayes import GaussianNB
import mlpy
import pylab
import matplotlib.pyplot as plt
import cPickle
import neurolab
import pandas as pd
from datetime import datetime

class RockPandas:
    def __init__(self):
        # self.train_file = 'train.csv'
        # self.test_file = 'test.csv'
        self.train_file = 'cv.train'
        self.test_file = 'cv.test'


    #==========Start ===========Pandas Seasonal Naive =======================
    def apply_format_id(self, df_row):
        return '{}_{}_{}'.format(df_row['Store'], df_row['Dept'], df_row['Date_Str'])

    def apply_fake_id_test(self, df_row):
    	# week = int(df_row['weeks']) + 2
    	# if week > 52:
    	# 	week =  week - 52
        return '{}_{}_{}_{}'.format(df_row['Store'], df_row['Dept'], (df_row['year'] - 1), df_row['weeks'])

    def apply_fake_id_train(self, df_row):
        return '{}_{}_{}_{}'.format(df_row['Store'], df_row['Dept'], df_row['year'], df_row['weeks'])


    def pandasSeasonalNaive(self):
        df_train = pd.read_csv(self.train_file)
        df_test = pd.read_csv(self.test_file)
        
        #Train
        df_train['Date_Str'] = df_train['Date']
        df_train['Date'] = pd.to_datetime(df_train['Date'])
        df_train = df_train.set_index(['Date'])
        
        # df_train['_id'] = df_train['Store'].astype(str) +'_' \
        #                         + df_train['Dept'].astype(str) +'_'+ df_train['Date_Str']

        df_train['_id'] = df_train.apply(self.apply_format_id, axis = 1)
        df_train['year'] =  df_train.index.year
        df_train['weeks'] = df_train.index.weekofyear
        df_train['fakeId'] = df_train.apply(self.apply_fake_id_train, axis = 1)

        #Test
        df_test['Date_Str'] = df_test['Date']
        df_test['Date'] = pd.to_datetime(df_test['Date'])
        df_test = df_test.set_index(['Date'])

        df_test['_id'] = df_test.apply(self.apply_format_id, axis = 1)
      	df_test['year'] =  df_test.index.year
        df_test['weeks'] = df_test.index.weekofyear
        df_test['fakeId'] = df_test.apply(self.apply_fake_id_test, axis = 1)
        
        
        merged = pd.merge(df_test, df_train, left_index=True, how='left', \
                                                        left_on='fakeId', right_on='fakeId', suffixes=('_test', '_train'))
        
        df_result = merged[['Date_Str_test', 'Weekly_Sales']]
        # df_result = merged[['_id_test', 'Weekly_Sales']]
        # print df_result
        df_result['Weekly_Sales'] = df_result['Weekly_Sales'].fillna(0)
        # print df_result
        
        df_result.to_csv('cv.seasonal_naive.csv', sep=',', index=False)
    #==========END ===========Pandas Seasonal Naive =======================


if  __name__=='__main__':
    pd.set_option('display.mpl_style', 'default')
    # pd.set_option('display.line_width', 5000) 
    pd.set_option('display.height', 500)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 60)
    rock = RockPandas()
    rock.pandasSeasonalNaive()