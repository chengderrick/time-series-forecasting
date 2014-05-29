import sys
from pprint import pprint
import numpy
import scipy
import math
from sklearn import linear_model, svm, tree
from sklearn.naive_bayes import GaussianNB
import mlpy
import pylab
import matplotlib.pyplot as plt
import cPickle
import neurolab
import pandas as pd
from datetime import datetime

# Class for collection of dept sales docs, 
# each store has many depts and each dept has a doc containing its weekly sales records
"""Global variables"""
TRAIN_SIZE = 143
TEST_SIZE = 39

TRAIN_START = '2010-02-05'
TRAIN_END = '2012-10-26'
TEST_START = '2012-11-02'
TEST_END = '2013-07-26'

class WalmartSalesColl:
    def __init__(self, fileName, fileType, crossValidate, givenLines):
        self.fileName = fileName
        self.fileType = fileType
        self.crossValidate = crossValidate
        self.givenLines = givenLines
        self.lines = self.readFileLines()  # raw file lines, if it's in cross-validation mode, lines = givenLines
        self.recordNumber = len(self.lines)
        self.allDeptdocs = self.buildAllDeptSalesDocs()  # all dept docs for the time being

        self.gDataFrame = self.fetchGlobalDf()

        # self.storeDocs = {}  # dict {1 : store object, 2: store object} and store doc contains dept objects
        # self.storeDocsList = [] # list has order
        # self.buildStoreSalesDocs()

    def fetchGlobalDf(self):
        df = pd.read_csv(self.fileName)
        df['Date_Str'] = df['Date']
        df['Date'] = pd.to_datetime(df['Date'])
        if self.fileType == 'test': 
            df['_id'] = df['Store'].map(str)+'_' + df['Dept'].map(str) + '_' + df['Date_Str'].map(str)
            # df['Weekly_Sales'] = 0
        indexed_df = df.set_index(['Date'])
        return indexed_df

    def buildAllDeptSalesDocs(self):
        storeId = 1
        deptId = 1
        deptSalesLines = []
        allDeptdocs = []
        for line in self.lines:
            lineInfo = line.split(',', 2)
            #check departmet change
            if int(lineInfo[1])!= deptId or int(lineInfo[0]) !=storeId:
                deptDoc = DeptDoc(deptSalesLines, storeId, deptId, self.fileType)
                #Append to global list, every dept will have a doc
                allDeptdocs.append(deptDoc)
                storeId = int(lineInfo[0])
                deptId = int(lineInfo[1])
                #clear list
                deptSalesLines = []
            
            deptSalesLines.append(line)

        # feed last doc
        deptDoc = DeptDoc(deptSalesLines, storeId, deptId, self.fileType)
        allDeptdocs.append(deptDoc)

        return allDeptdocs

    def buildStoreSalesDocs(self):
        storeId = 1
        storeSalesLines = []
        for line in self.lines:
            lineInfo = line.split(',', 1)
            #check store change
            if int(lineInfo[0]) !=storeId:
                storeDoc = StoreDoc(storeSalesLines, storeId, self.fileType)
                self.storeDocs[storeId] = storeDoc # Append to global dict, every store will have a doc
                self.storeDocsList.append(storeDoc) # the same but ordered lsit

                storeId = int(lineInfo[0])
                storeSalesLines = []  # clear list
            
            storeSalesLines.append(line)

        # feed last doc
        storeDoc = StoreDoc(storeSalesLines, storeId, self.fileType)
        self.storeDocs[storeId] = storeDoc
        self.storeDocsList.append(storeDoc)


    def readFileLines(self):
        if self.crossValidate:
            return self.givenLines # given train or test lines for cross validation

        f = open(self.fileName)
        rawLines = f.readlines()
        rawLines.pop(0)
        f.close()

        return rawLines

    def saveResultList(self, results, outputFile):
        f = open(outputFile,'w+')
        for item in results:
            f.write("%s\n" % item)

    def outputForecastResult(self, results, outputFile):
        f = open(outputFile,'w+')
        # for item in forecastResults:
        #     f.write("%s\n" % item)
        i = 0
        writeLines = []
        for line in self.lines:
            testlist = line.split(',')
            resultToFile = testlist[0]+'_'+testlist[1]+'_'+testlist[2]+','+str(float(results[i]*100)/100)
            writeLines.append(resultToFile)
            i+=1

        f.write("\n".join(writeLines))

    def plotDeptSalesData():
        test =0
        #TODO: plot the Dept sale graph
        # Moving average gives trend, and deseasonalize the data, seasonality index

class StoreDoc:
    def __init__(self, storeSalesLines, storeId, fileType):
        self.storeId = storeId
        self.storeSalesLines = storeSalesLines
        self.recordNumber = len(storeSalesLines)
        self.fileType = fileType
        
        self.deptDocs = {}  # dict
        self.deptDocsList = [] # the same but ordered
        self.buildDeptDocsOfStore()

        self.storeSalesInfo = self.createStoreSalesInfo()
        self.avgStoreSales = self.getAvgSalesOfStore()

    def createStoreSalesInfo(self):
        if self.fileType == 'test':
            return []

        salesLineInfo = []
        for line in self.storeSalesLines:
            salesInfoSplit = line.split(',')
            salesLineInfo.append(salesInfoSplit)  # create a matrix storing each info of a store sales line
        return salesLineInfo

    def getAvgSalesOfStore(self):
        if self.fileType == 'train':
            dt = numpy.dtype(float)
            salesColumn = numpy.array(self.storeSalesInfo)[:,3].astype(dt)
            return numpy.mean(salesColumn)
            
        

    def buildDeptDocsOfStore(self):
        deptId = 1
        deptSalesLines = []

        for salesLine in self.storeSalesLines:
            lineInfo = salesLine.split(',', 2)
            #check departmet change
            if int(lineInfo[1])!= deptId:
                deptDoc = DeptDoc(deptSalesLines, self.storeId, deptId, self.fileType)
                #Append to global list, every dept will have a doc
                self.deptDocs[deptId] = deptDoc
                self.deptDocsList.append(deptDoc)
                deptId = int(lineInfo[1])
                #clear list
                deptSalesLines = []

            deptSalesLines.append(salesLine)

        # feed last doc
        deptDoc = DeptDoc(deptSalesLines, self.storeId, deptId, self.fileType)
        self.deptDocs[deptId] = deptDoc
        self.deptDocsList.append(deptDoc)

class DeptDoc:
    def __init__(self, deptSalesLines, storeId, deptId, fileType):
        self.deptSalesLines = deptSalesLines
        self.fileType = fileType
        self.deptSalesInfo = self.createDeptSalesInfo() # [[],[]]create a matrix storing each info of a dept sales line
        self.storeId = storeId
        self.deptId = deptId
        self.recordNumber = len(self.deptSalesLines)
        # self.salesColumn = self.getSalesColumn()
        # self.avgDeptSales = self.getAvgSalesOfDept()

    def createDeptSalesInfo(self):
        weeklyInfo = []
        for salesLine in self.deptSalesLines:
            salesSplit = salesLine.split(',')
            weeklyInfo.append(salesSplit)
        return weeklyInfo

    def getSalesColumn(self):
        if self.fileType == 'test':
            return []
        dt = numpy.dtype(float)
        salesColumn = numpy.array(self.deptSalesInfo)[:,3].astype(dt)

        return salesColumn

    def getAvgSalesOfDept(self):
        if self.fileType == 'train':
            return numpy.mean(self.salesColumn)

    def fetchTrainDoc(self, trainSalesColl):
        if self.deptId in trainSalesColl.storeDocs[self.storeId].deptDocs:
            trainDeptDoc = trainSalesColl.storeDocs[self.storeId].deptDocs[self.deptId]
        else:
            trainDeptDoc = None #TODO: Even though there is no cooresponding one in train, find a simialr one from other stores
        
        return trainDeptDoc

    def build_df_query(self, coll, field, value):
        return coll.gDataFrame[field] == value





    #===================== Pandas ES ========================
    def pandasES(self, testSalesColl, trainSalesColl, df_dates_all):
        print 'store', self.storeId, 'dept', self.deptId
        
        where_train_store = self.build_df_query(trainSalesColl, 'Store', self.storeId)
        where_train_dept = self.build_df_query(trainSalesColl, 'Dept', self.deptId)
        df_train = trainSalesColl.gDataFrame[where_train_store & where_train_dept]

        # start = datetime(2010, 2, 5)
        # end = datetime(2012, 10, 26)
        # print df_train.ix[start:end]
        print df_train['2010-2-5':'2010-3-6']

        return 
        where_test_store = self.build_df_query(testSalesColl, 'Store', self.storeId)
        where_test_dept = self.build_df_query(testSalesColl, 'Dept', self.deptId)
        df_test = testSalesColl.gDataFrame[where_test_store & where_test_dept]

        

        if df_train.empty:
            testSalesColl.gDataFrame.loc[(where_test_store & where_test_dept), 'Weekly_Sales'] \
                                = trainSalesColl.gDataFrame[where_train_store]['Weekly_Sales'].mean()

        else:
            testSalesColl.gDataFrame.loc[(where_test_store & where_test_dept), 'Weekly_Sales'] \
                                = df_train['Weekly_Sales'].mean()


    #===========END========== Pandas ES ========================

    #=============== Pandas DTW ================================
    # ---------  This block is for Dynamic Time Warping  ---------
    def decideDTWWindowSize(self, trainLength, forecastSteps):
        # Note: train dept can be NONE here, need to fix the fetch part first
        space = trainLength - forecastSteps
        
        if space/4 > 0:  # Normal situation
            return space/4
        else:  # There are more test than train or train data is not sufficient, fall back to linear regression
            return -1

    def forecastDTW(self, trainSalesColl):
        trainDeptDoc = self.fetchTrainDoc(trainSalesColl)
        print 'store', self.storeId, 'dept', self.deptId
        #TODO: Temp:
        if trainDeptDoc == None:
            trainLength = 0
        else:
            trainLength = trainDeptDoc.recordNumber
        # Temp END
        
        forecastSteps = self.recordNumber
        #trainLength = trainDeptDoc.recordNumber  #TODO: Temp, should be uncommented when related dept found
        windowSize = self.decideDTWWindowSize(trainLength, forecastSteps)
        
        # When windowSize is -1, use linear regression instead
        if windowSize == -1:
            return self.forecastRegression(trainSalesColl)

        salesColumn = trainDeptDoc.salesColumn
        match = mlpy.dtw_subsequence(salesColumn[-windowSize: ], salesColumn[ :-forecastSteps])
        matchPoint = match[2][1][-1]
        forecastResults = salesColumn[matchPoint + 1 : matchPoint + forecastSteps + 1]
        return forecastResults.tolist()


    #============END Pandas DTW ================================




    #=================Pandas Multipe Linear Regression ===========================
    def trainClassifier(self, trainData, targetData):
        classifier = linear_model.LinearRegression()
        # classifier = linear_model.LassoLars(alpha=.1)
        # classifier = svm.SVR()
        # classifier = tree.DecisionTreeRegressor()
        # classifier = GaussianNB()

        classifier.fit(trainData, targetData)
        return classifier

    def decideRegressionWindowSize(self, recordNumber):
        if recordNumber/5 > 0:
            return int(recordNumber/5)
        elif recordNumber/2 > 0:
            return int(recordNumber/2)
        else:
            return 0

    def pandasRegression(self, testSalesColl, trainSalesColl, df_dates_all):
        print 'store', self.storeId, 'dept', self.deptId
        
        where_train_store = self.build_df_query(trainSalesColl, 'Store', self.storeId)
        where_train_dept = self.build_df_query(trainSalesColl, 'Dept', self.deptId)
        df_train = trainSalesColl.gDataFrame[where_train_store & where_train_dept]
        
        where_test_store = self.build_df_query(testSalesColl, 'Store', self.storeId)
        where_test_dept = self.build_df_query(testSalesColl, 'Dept', self.deptId)
        df_test = testSalesColl.gDataFrame[where_test_store & where_test_dept]

        if df_train.empty:
            df_avg = df_test[['_id']]
            df_avg['Weekly_Sales'] = trainSalesColl.gDataFrame[where_train_dept]['Weekly_Sales'].mean()
            return df_avg
        
        merged = pd.merge(df_dates_all, df_train, left_index=True, right_index=True, how='left', \
                                                                    suffixes=('_all', '_train'))
        # Fill any missing data
        merged[TRAIN_START:TRAIN_END] = merged[TRAIN_START:TRAIN_END].interpolate()
        merged[TRAIN_START:TRAIN_END] = merged[TRAIN_START:TRAIN_END].fillna(method='bfill')
    
        windowSize = self.decideRegressionWindowSize(TRAIN_SIZE)

        dt = numpy.dtype(float)
        salesColumn = numpy.array(merged[TRAIN_START:TRAIN_END]['Weekly_Sales']).astype(dt)
        
        trainList = [] # all training data e.g. [ [], [] ]
        targetList = [] # numberic targets e.g. [ ]
        head = -1

        # Generate training data from train
        while abs(head) + windowSize <= TRAIN_SIZE:
            tail = head - windowSize
            trainInstance = salesColumn[tail: head]
            trainList.append(trainInstance)
            targetList.append(salesColumn[head])
            head -= 1

        # Construct initial model
        trainList = numpy.array(trainList)
        targetList = numpy.array(targetList)

        classifier = self.trainClassifier(trainList, targetList)
        #print('Coefficients: \n', classifier.coef_)

        # Make forecasting and add new instance into model
        forecastResults =[]
        for i in range (0, TEST_SIZE):
            newTrainInstance = salesColumn[-windowSize: ]

            target = classifier.predict(newTrainInstance)
            forecastResults.append(target)
            salesColumn = numpy.append(salesColumn, target)
            # Feed new instance
            trainList = numpy.vstack((trainList, newTrainInstance))
            targetList = numpy.append(targetList, target)
            classifier = self.trainClassifier(trainList, targetList)
        
        merged.loc[TEST_START:TEST_END, 'Weekly_Sales'] = forecastResults

        merged_test = pd.merge(df_test, merged, left_index=True, right_index=True, \
                                                how='left', suffixes=('_test', '_all'))
        return merged_test[['_id', 'Weekly_Sales']]

    #====================END========== andas Multipe Linear Regression ==================



    #===================== Pandas Mean ========================
    def pandasMean(self, testSalesColl, trainSalesColl, df_dates_all):
        print 'store', self.storeId, 'dept', self.deptId
        
        where_train_store = self.build_df_query(trainSalesColl, 'Store', self.storeId)
        where_train_dept = self.build_df_query(trainSalesColl, 'Dept', self.deptId)
        df_train = trainSalesColl.gDataFrame[where_train_store & where_train_dept]
        
        where_test_store = self.build_df_query(testSalesColl, 'Store', self.storeId)
        where_test_dept = self.build_df_query(testSalesColl, 'Dept', self.deptId)
        df_test = testSalesColl.gDataFrame[where_test_store & where_test_dept]

        # print df_train['Weekly_Sales'].mean()

        if df_train.empty:
            df_avg = df_test[['_id']]
            # get the mean of all the same dept cross all stores
            df_avg['Weekly_Sales'] = trainSalesColl.gDataFrame[where_train_store]['Weekly_Sales'].mean()
            return df_avg

        else:
            df_avg = df_test[['_id']]
            df_avg['Weekly_Sales'] = df_train['Weekly_Sales'].mean()
            return df_avg

    #===========END========== Pandas Mean ========================


    #==========END ===========Pandas Naive =======================
    def pandasNaive(self, trainSalesColl, df_dates_all):
        print 'store', self.storeId, 'dept', self.deptId

        trainDeptDoc = self.fetchTrainDoc(trainSalesColl)
        test_lines = self.deptSalesInfo

        df_test = pd.DataFrame(test_lines, columns=['Store','Dept','Date_Str', 'IsHoliday'])
        df_test['Date'] = pd.to_datetime(df_test['Date_Str'])
        df_test['_id'] = df_test['Store'].map(str)+'_' + df_test['Dept'].map(str) \
                                                + '_' + df_test['Date_Str'].map(str)
        if trainDeptDoc == None:
            df_avg = df_test[['_id']]
            df_avg['Weekly_Sales'] = trainSalesColl.storeDocs[self.storeId].avgStoreSales
            return df_avg

        train_lines = trainDeptDoc.deptSalesInfo        
        df_train = pd.DataFrame(train_lines, columns=['Store','Dept','Date_Str','Weekly_Sales','IsHoliday'])
        df_train['Date'] = pd.to_datetime(df_train['Date_Str'])
        # pprint(df_train)
        

        merged = pd.merge(df_dates_all, df_train, on='Date', how='left', suffixes=('_all', '_train'))
        merged_ffill = merged.ffill()
        merged_test = pd.merge(df_test, merged_ffill, on='Date', how='left', suffixes=('_test', '_all'))

        # pprint(merged_test[['_id', 'Weekly_Sales']])
        return merged_test[['_id', 'Weekly_Sales']]

    #=====================Pandas Naive =======================

    #====
    #after pandas funcs removed
    #====

class Validation:
    def __init__(self, trainFile):
        self.trainFile = trainFile
        self.trainColl = WalmartSalesColl(trainFile, 'train', False, [])

        self.trainLines = []
        self.holdoutLines = []
        self.holdout()   # populate above lists

        self.trainDataColl = WalmartSalesColl(self.trainFile, 'train', True, self.trainLines)
        self.holdoutDataColl = WalmartSalesColl(self.trainFile, 'train', True, self.holdoutLines)

        self.forecastResultsDict = {}

        # print "self.trainLines-------------"
        # pprint(self.trainLines)
        # print "self.holdoutLinesholdoutLines-------------"
        # pprint(self.holdoutLines)

    def holdout(self):
        for deptDoc in self.trainColl.allDeptdocs:
            holdoutPoint = self.getHoldoutPoint(deptDoc.recordNumber)
            self.trainLines += deptDoc.deptSalesLines[ : deptDoc.recordNumber - holdoutPoint]
            self.holdoutLines += deptDoc.deptSalesLines[-holdoutPoint : ]
            
    def getHoldoutPoint(self, recordNumber):
        if recordNumber/4 > 0:
            return recordNumber/4
        # elif recordNumber/4 > 0:
        #      return recordNumber/4
        elif recordNumber/2 > 0:
            return recordNumber/2
        else:
            return recordNumber

    def validate(self):
        forecastResults = []
        # featureColl = feature.FeaturesColl('features.csv')

        for testDeptDoc in self.holdoutDataColl.allDeptdocs:
            # print "test deptSalesLines:"
            # pprint(testDeptDoc.deptSalesLines)
            if testDeptDoc.storeId == 1 and testDeptDoc.deptId == 1:
                deptForecast = []
                # deptForecast = testDeptDoc.forecastRegression(self.trainDataColl)
                # deptForecast = testDeptDoc.forecastFeaturedRegression(self.trainDataColl, featureColl)
                # deptForecast = testDeptDoc.forecastDTW(self.trainDataColl)
                deptForecast = testDeptDoc.forecastANN(self.trainDataColl)
                forecastResults += deptForecast
                self.forecastResultsDict[str(testDeptDoc.storeId) + '_' + str(testDeptDoc.deptId)] = deptForecast

            # forecastResults += testDeptDoc.forecastDTW(trainDataColl)
        # print "forecastResults:"
        # pprint(forecastResults)

        # to unccomnet
        # WMAE = self.evaluation(forecastResults, self.holdoutDataColl) # Evaluation the results against our metric
        # print 'Final WMAE is', WMAE
        # return WMAE

    # Method evaluation
    # forecast: [] 
    # realColl: instance of WalmartSalesColl contains actual sales
    def evaluation(self, forecast, realColl):
        if len(forecast) != realColl.recordNumber:
            print "Something wrong, the forecast is not complete"
            return
        WMAESum = 0
        weightSum = 0
        i = 0
        for storeDoc in realColl.storeDocsList:
            for deptDoc in storeDoc.deptDocsList:
                print 'storeID', deptDoc.storeId, 'deptID', deptDoc.deptId
                for splitInfo in deptDoc.deptSalesInfo:
                    weightedAbsError = 0
                    actualValue = float(splitInfo[3])
                    isHoliday = True if splitInfo[4].strip() == 'TRUE' else False
                    weightedAbsError = 5 * (abs(actualValue - forecast[i])) if isHoliday else abs(actualValue - forecast[i])
                    WMAESum += weightedAbsError
                    weightSum +=  5 if isHoliday else 1
                    i += 1
        
        print 'test data size', len(forecast)
        return float(WMAESum)/weightSum

class Helper:
    def getFullDateRange(self):
        start = datetime(2010, 2, 5)
        train_end = datetime(2012, 10, 26)
        test_start = datetime(2012, 11, 2)
        test_end = datetime(2013, 7, 26)

        df_dates_all = pd.date_range(start, test_end, freq="W-FRI")
        df_dates_all = pd.DataFrame(pd.Series(df_dates_all), columns=['Date'])
        df_dates_all = df_dates_all.set_index('Date')
        return df_dates_all

    def plotTestData(self, testFile):
        testLines = self.readFileLines(testFile)
        deptDict = {}
        counter = 1
        deptFlag = '1_1'
        for line in testLines:
            deptInfo = line.split(',')
            if deptFlag != deptInfo[0] +'_'+ deptInfo[1]: 
                counter += 1
                deptFlag = deptInfo[0] +'_'+ deptInfo[1]
            deptDict[counter] = deptDict.get(counter, 0) + 1
            
        deptIds = deptDict.keys()
        deptIds.sort()
        deptNumber = []

        for id in deptIds:
            if id == 501:
                break
            deptNumber.append(deptDict[id])

        # pprint(deptIds)
        # pprint(deptNumber)
        self.drawGraph(deptIds[0:500], deptNumber, 'Test')

    def plotHoldoutData(self):
        validator = Validation('train.csv')
        lines = validator.holdoutLines
        deptDict = {}
        counter = 1
        deptFlag = '1_1'
        for line in lines:
            deptInfo = line.split(',')
            if deptFlag != deptInfo[0] +'_'+ deptInfo[1]: 
                counter += 1
                deptFlag = deptInfo[0] +'_'+ deptInfo[1]
            deptDict[counter] = deptDict.get(counter, 0) + 1

        deptIds = deptDict.keys()
        deptIds.sort()
        deptNumber = []
        for id in deptIds:
            if id == 501:
                break
            deptNumber.append(deptDict[id])

        # print deptIds
        self.drawGraph(deptIds[0:500], deptNumber, 'Holdout')


    def readFileLines(self, testFileName):
        f = open(testFileName)
        lines = f.readlines()
        lines.pop(0)
        f.close()

        return lines

    def drawGraph(self, xData, yData, title):
        pylab.title(title + ' dept data distribution')
        pylab.xlabel('Dept')
        pylab.ylabel('Number of records in the dept')

        pylab.figure(1)
        pylab.plot(xData, yData)
        pylab.show() 

    def plotTestData2(self, testFile):
        testLines = self.readFileLines(testFile)
        deptList = []
        counter = 1 # dept no in int
        deptFlag = '1_1'
        for line in testLines:
            deptInfo = line.split(',')
            if deptFlag != deptInfo[0] +'_'+ deptInfo[1]:
                if counter == 100:
                    break 
                counter += 1
                deptFlag = deptInfo[0] +'_'+ deptInfo[1]
            deptList.append(counter)

        # pprint(deptIds)
        # pprint(deptNumber)
        print counter, 'haj'
        self.histo(counter, deptList, 'Test')

    def plotHoldoutData2(self):
        validator = Validation('train.csv')
        lines = validator.holdoutLines
        deptList = []
        counter = 1 # dept no in int
        deptFlag = '1_1'
        for line in lines:
            deptInfo = line.split(',')
            if deptFlag != deptInfo[0] +'_'+ deptInfo[1]:
                if counter == 100:
                    break 
                counter += 1
                deptFlag = deptInfo[0] +'_'+ deptInfo[1]
            deptList.append(counter)

        # pprint(deptIds)
        # pprint(deptNumber)
        print counter, 'haj'
        self.histo(counter, deptList, 'Holdout')

    def histo(self, nbins, deptList, title):
        plt.hist(deptList, bins=nbins, color='blue')
        plt.title(title)
        plt.show()

    def countHoliday(self, lines):
        allCount = len(lines);
        print allCount
        holidayCount = 0
        for line in lines:
            words = line.split(',')
            if words[-1].strip() == 'TRUE':
                holidayCount += 1

        print float(holidayCount)/allCount

    def dumpWalmartSalesCollToDisk(self):
        trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
        trainCollFile = open('train-coll.obj', 'w') 
        cPickle.dump(trainSalesColl, trainCollFile) 

    def loadWalmartSalesCollFromDisk(self):
        trainCollFile = open('train-coll.obj', 'r') 
        trainSalesColl = pickle.load(trainCollFile)
        print len(trainSalesColl.storeDocs[1].deptDocs[1].deptSalesLines)


    def plotDeptSalesFigure(self, contextDeptDoc, realDeptDoc, forecastSalesRecords):
        if len(realDeptDoc.salesColumn) != len(forecastSalesRecords):
            print "Something wrong is happen, please check"
            return

        
        numberRecords = len(contextDeptDoc.salesColumn) + len(realDeptDoc.salesColumn)
        xScale = range(0, numberRecords)
        xScaleForecast = range(len(contextDeptDoc.salesColumn), numberRecords)

        realGraph = contextDeptDoc.salesColumn.tolist() + realDeptDoc.salesColumn.tolist()
        print len(realGraph), len(forecastSalesRecords), 'test new'

        plt.plot(xScale, realGraph)
        plt.plot(xScaleForecast, forecastSalesRecords)

        plt.show()


if  __name__=='__main__':
    pd.set_option('display.mpl_style', 'default')
    # pd.set_option('display.line_width', 5000) 
    pd.set_option('display.height', 500)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 60) 

    #--------- Pandas MLR START
    trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])
    helper = Helper()

    df_dates_all = helper.getFullDateRange()
    forecastResults = pd.DataFrame(None, columns=['_id', 'Weekly_Sales'])
    for deptDoc in testSalesColl.allDeptdocs:
        df_result = deptDoc.pandasRegression(testSalesColl, trainSalesColl, df_dates_all)
        forecastResults = forecastResults.append(df_result)
    print "OF cousre, Done"
    # pprint(testSalesColl.gDataFrame)
    forecastResults.to_csv('linear_regression.csv', sep=',', index=False)

    # pprint(forecastResults)
    # testSalesColl.saveResultList(forecastResults, 'ANNRescue.txt')
    # testSalesColl.outputForecastResult(forecastResults, 'finalResultsANN.csv')
    #--------- Pandas MLR END
    




    # #--------- Pandas ES START
    # trainSalesColl = WalmartSalesColl('mock.train', 'train', False, [])
    # testSalesColl = WalmartSalesColl('mock.test', 'test', False, [])
    # helper = Helper()
    # df_dates_all = helper.getFullDateRange()
    # forecastResults = pd.DataFrame(None, columns=['_id', 'Weekly_Sales'])
    # for deptDoc in testSalesColl.allDeptdocs:
    #     df_result = deptDoc.pandasES(testSalesColl, trainSalesColl, df_dates_all)
    #     forecastResults = forecastResults.append(df_result)
    # print "OF cousre"
    # # pprint(testSalesColl.gDataFrame)
    # # forecastResults.to_csv('es.csv', sep=',', index=False)

    # # pprint(forecastResults)
    # # testSalesColl.saveResultList(forecastResults, 'ANNRescue.txt')
    # # testSalesColl.outputForecastResult(forecastResults, 'finalResultsANN.csv')
    # #--------- Pandas ES END


    # #--------- Pandas MEAN START
    # trainSalesColl = WalmartSalesColl('mock.train', 'train', False, [])
    # testSalesColl = WalmartSalesColl('mock.test', 'test', False, [])
    # helper = Helper()
    # df_dates_all = helper.getFullDateRange()
    # forecastResults = pd.DataFrame(None, columns=['_id', 'Weekly_Sales'])
    # for deptDoc in testSalesColl.allDeptdocs:
    #     df_result = deptDoc.pandasMean(testSalesColl, trainSalesColl, df_dates_all)
    #     forecastResults = forecastResults.append(df_result)

    # forecastResults.to_csv('mean_by_store.csv', sep=',', index=False)
    # # pprint(forecastResults)
    # # testSalesColl.saveResultList(forecastResults, 'ANNRescue.txt')
    # # testSalesColl.outputForecastResult(forecastResults, 'finalResultsANN.csv')
    # #--------- Pandas MEAN END

    # #--------- Pandas NAIVE START
    # # train collection
    # trainSalesColl = WalmartSalesColl('mock.train', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('mock.test', 'test', False, [])
    # helper = Helper()
    # df_dates_all = helper.getFullDateRange()
    # forecastResults = pd.DataFrame(None, columns=['_id', 'Weekly_Sales'])
    # for deptDoc in testSalesColl.allDeptdocs:
    #     df_result = deptDoc.pandasNaive(trainSalesColl, df_dates_all)
    #     forecastResults = forecastResults.append(df_result)

    # forecastResults.to_csv('mean.csv', sep=',', index=False)
    # # pprint(forecastResults)
    # # testSalesColl.saveResultList(forecastResults, 'ANNRescue.txt')
    # # testSalesColl.outputForecastResult(forecastResults, 'finalResultsANN.csv')
    # #--------- Pandas NAIVE END


    # #--------- ANN START
    # # train collection
    # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    # forecastResults = []
    # for deptDoc in testSalesColl.allDeptdocs:
    #     forecastResults += deptDoc.forecastANN(trainSalesColl)
    
    # # pprint(forecastResults)
    # testSalesColl.saveResultList(forecastResults, 'ANNRescue.txt')
    # testSalesColl.outputForecastResult(forecastResults, 'finalResultsANN.csv')
    # #--------- ANN END


    # #--------- START helper
    # helper = Helper()
    # helper.dumpWalmartSalesCollToDisk()
    # #--------- START helper


    # #--------- START Plot Dept figure
    # # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # validator = Validation('train.csv')
    # validator.validate()
    # helper = Helper()
    # storeId = 1
    # deptId = 1
    # helper.plotDeptSalesFigure(validator.trainDataColl.storeDocs[storeId].deptDocs[deptId], validator.holdoutDataColl.storeDocs[storeId].deptDocs[deptId], validator.forecastResultsDict[str(storeId)+'_'+str(deptId)])
    # #--------- END Plot Dept figure



    # #--------- START Holiday Dists
    # validator = Validation('train.csv')
    # # validator.validate()
    # # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])
    # helper = Helper()
    # # helper.countHoliday(testSalesColl.lines)
    # helper.countHoliday(validator.holdoutLines)
    
    # #--------- END Holiday Dists


    # #--------- START validation
    # validator = Validation('mock.train')
    # validator.validate()
    # #--------- END validation


    # #--------- Featured Multipe Linear Regression START
    # featureColl = feature.FeaturesColl('features.csv')

    # # train collection /train.csv/test.csv/mock.train/mock.test
    # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    # forecastResults = []
    # for deptDoc in testSalesColl.allDeptdocs:
    #     forecastResults += deptDoc.forecastFeaturedRegression(trainSalesColl, featureColl)
    
    # testSalesColl.outputForecastResult(forecastResults, 'TheResults.csv')
    # #--------- Featured Multipe Linear Regression END


    # helper = Helper()
    # # helper.plotTestData2('test.csv')
    # helper.plotHoldoutData2()




    #--------- Exponential Smoothing START
    # # train collection
    # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    # forecastResults = []
    # for deptDoc in testSalesColl.allDeptdocs:
    #     forecastResults += deptDoc.forecastExponentialSmoothing(trainSalesColl)

    # pprint(len(forecastResults))
    # testSalesColl.outputForecastResult(forecastResults, 'finalResultsFromDTW.csv')
    #--------- Exponential Smoothing END


    # #--------- Multiple Regression START
    # # train collection
    # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    # forecastResults = []
    # for deptDoc in testSalesColl.allDeptdocs:
    #     forecastResults += deptDoc.forecastRegression(trainSalesColl)
    
    # # pprint(forecastResults)
    # testSalesColl.outputForecastResult(forecastResults, 'finalResults.csv')
    # #--------- Multiple Regression END


    # #--------- ensemble Regression + DTW START
    # # train collection
    # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    # forecastResults = []
    # for deptDoc in testSalesColl.allDeptdocs:
    #     forecastResults1 = deptDoc.forecastRegression(trainSalesColl)
    #     forecastResults2 = deptDoc.forecastDTW(trainSalesColl)
    #     forecastResults += [x/2.0 for x in map(sum, zip(forecastResults1, forecastResults2))]
    
    # # pprint(forecastResults)
    # testSalesColl.outputForecastResult(forecastResults, 'finalResults.csv')
    # #--------- ensemble Regression + DTW END


    # #--------- DTW START
    # # train collection
    # trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # # test collection
    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    # forecastResults = []
    # for deptDoc in testSalesColl.allDeptdocs:
    #     forecastResults += deptDoc.forecastDTW(trainSalesColl)
    
    # pprint(len(forecastResults))
    # testSalesColl.outputForecastResult(forecastResults, 'finalResultsFromDTW.csv')
    # #--------- DTW END






    # testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])
    # f = open('testDeptRecordsNumber.csv','w+')
    # for deptDoc in testSalesColl.allDeptdocs:
    #     print >>f, deptDoc.storeId, deptDoc.deptId, deptDoc.recordNumber
    
     
