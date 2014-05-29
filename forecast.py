import sys
from pprint import pprint
import numpy
import scipy
import math
from sklearn import linear_model, svm, tree
from sklearn.naive_bayes import GaussianNB
import mlpy
import feature
import pylab
import matplotlib.pyplot as plt
import cPickle
import neurolab

# Class for collection of dept sales docs, 
# each store has many depts and each dept has a doc containing its weekly sales records
class WalmartSalesColl:
    def __init__(self, fileName, fileType, crossValidate, givenLines):
        self.fileName = fileName
        self.fileType = fileType
        self.crossValidate = crossValidate
        self.givenLines = givenLines
        self.lines = self.readFileLines()  # raw file lines, if it's in cross-validation mode, lines = givenLines
        self.recordNumber = len(self.lines)
        self.allDeptdocs = self.buildAllDeptSalesDocs()  # all dept docs for the time being

        self.storeDocs = {}  # dict {1 : store object, 2: store object} and store doc contains dept objects
        self.storeDocsList = [] # list has order
        self.buildStoreSalesDocs()

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
        self.salesColumn = self.getSalesColumn()
        self.avgDeptSales = self.getAvgSalesOfDept()

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

    # ----Featured-----  This block is for Featured Multipe Linear Regression ---------
    def fetchFeatures(self, salesWithDate, salesColumn, tail, head, featureColl):
        salesFeature = salesColumn[tail : head]
        dates = salesWithDate[tail : head][:,0]

        temperatureFeature = []
        fuelPriceFeature = []
        CPIFeature = []
        unemploymentFeature = []
        isHoliday = []
        for date in dates:
            featureOfDate = featureColl.storeFeatureDocs[self.storeId].featureVectors[date]
            if featureOfDate['Temperature'] != 'NA':
                temperatureFeature.append(float(featureOfDate['Temperature']))
            # if featureOfDate['Fuel_Price'] != 'NA':
            #     fuelPriceFeature.append(float(featureOfDate['Fuel_Price']))
            # if featureOfDate['CPI'] != 'NA':
            #     CPIFeature.append(float(featureOfDate['CPI']))
            # if featureOfDate['Unemployment'] != 'NA':
            #     unemploymentFeature.append(float(featureOfDate['Unemployment']))

        avgTemperature = sum(temperatureFeature) / (len(temperatureFeature) if len(temperatureFeature) > 0 else 1)
        # avgFuelPrice = sum(fuelPriceFeature) / (len(fuelPriceFeature) if len(fuelPriceFeature) > 0 else 1)
        # avgCPI = sum(CPIFeature) / (len(CPIFeature) if len(CPIFeature) > 0 else 1)
        # avgUnemployment = sum(unemploymentFeature) / (len(unemploymentFeature) if len(unemploymentFeature) > 0 else 1)

        trainFeatureInstance = salesFeature.tolist()

        trainFeatureInstance.append(avgTemperature)
        # trainFeatureInstance.append(avgFuelPrice)
        # trainFeatureInstance.append(avgCPI)
        # trainFeatureInstance.append(avgUnemployment)


        trainFeatureInstance = map(float, trainFeatureInstance)
        # pprint(trainFeatureInstance)

        return trainFeatureInstance

    def forecastFeaturedRegression(self, trainSalesColl, featureColl):
        trainDeptDoc = self.fetchTrainDoc(trainSalesColl)
        # pprint(trainDeptDoc.deptSalesInfo)
        if trainDeptDoc == None:
            return [ trainSalesColl.storeDocs[self.storeId].avgStoreSales ] * self.recordNumber  # if test dept is not in training data
        
        windowSize = self.decideRegressionWindowSize(trainDeptDoc.recordNumber)
        
        if windowSize == 0:
            return [ trainSalesColl.storeDocs[self.storeId].avgStoreSales ] * self.recordNumber
        
        salesWithDate = numpy.array(trainDeptDoc.deptSalesInfo)[:,[2,3]]
        salesColumn = salesWithDate[:,1].astype('float')

        trainList = [] # all training data e.g. [ [], [] ]
        targetList = [] # numberic targets e.g. [ ]
        head = -1
        # Generate training data from train
        while abs(head) + windowSize <= trainDeptDoc.recordNumber:
            tail = head - windowSize
            trainInstance = self.fetchFeatures(salesWithDate, salesColumn, tail, head, featureColl)
            trainList.append(trainInstance)
            targetList.append(salesColumn[head])
            head -= 1


        # Construct initial model
        trainList = numpy.array(trainList)
        targetList = numpy.array(targetList)

        classifier = self.trainClassifier(trainList, targetList)

        # Make forecasting and add new instance into model
        forecastResults =[]
        for i in range (0, self.recordNumber):
            newTrainInstance = self.fetchFeatures(salesWithDate, salesColumn, -windowSize, None, featureColl)
            # print newTrainInstance

            target = classifier.predict(newTrainInstance)
            # print 'target', target
            # supply train as the classification goes
            forecastResults.append(target)
            salesColumn = numpy.append(salesColumn, target)
            salesWithDate = numpy.vstack((salesWithDate, [self.deptSalesInfo[i][2], target]))  # get date from test
            
            # Feed new instance
            trainList = numpy.vstack((trainList, newTrainInstance))
            targetList = numpy.append(targetList, target)
            classifier = self.trainClassifier(trainList, targetList)
        print 'store', self.storeId, 'dept', self.deptId
        #pprint(forecastResults)
        return forecastResults 

    # ---------  This block is for Multipe Linear Regression  ---------
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
        # elif recordNumber/4 > 0:
        #     return int(recordNumber/4)
        elif recordNumber/2 > 0:
            return int(recordNumber/2)
        else:
            return 0

    def forecastRegression(self, trainSalesColl):
        if self.deptId in trainSalesColl.storeDocs[self.storeId].deptDocs:
            trainDeptDoc = trainSalesColl.storeDocs[self.storeId].deptDocs[self.deptId]
        else:
            trainDeptDoc = None

        if trainDeptDoc == None:
            return [ trainSalesColl.storeDocs[self.storeId].avgStoreSales ] * self.recordNumber  # if test dept is not in training data
        windowSize = self.decideRegressionWindowSize(trainDeptDoc.recordNumber)
        if windowSize == 0:
            return [ trainSalesColl.storeDocs[self.storeId].avgStoreSales ] * self.recordNumber

        dt = numpy.dtype(float)
        salesColumn = numpy.array(trainDeptDoc.deptSalesInfo)[:,3].astype(dt) #TODO can be replaced with the self.salesColumn
        trainList = [] # all training data e.g. [ [], [] ]
        targetList = [] # numberic targets e.g. [ ]
        head = -1

        # Generate training data from train
        while abs(head) + windowSize <= trainDeptDoc.recordNumber:
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
        for i in range (0, self.recordNumber):
            newTrainInstance = salesColumn[-windowSize: ]

            target = classifier.predict(newTrainInstance)
            forecastResults.append(target)
            salesColumn = numpy.append(salesColumn, target)
            # Feed new instance
            trainList = numpy.vstack((trainList, newTrainInstance))
            targetList = numpy.append(targetList, target)
            classifier = self.trainClassifier(trainList, targetList)

        print 'store', self.storeId, 'dept', self.deptId
        #pprint(forecastResults)
        return forecastResults

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

    # ---------- This block is for Simple Exponential Smoothing --------
    def calculateES(self, alpha, salesColumn, forecast0):
        esForecast = 0
        for i in range(1, len(salesColumn)+1):
            esForecast += alpha * math.pow((1 - alpha), i-1) * salesColumn[-i]

        esForecast += math.pow((1 - alpha), len(salesColumn)) * forecast0
        return esForecast

    def forecastExponentialSmoothing(self, trainSalesColl):
        trainDeptDoc = self.fetchTrainDoc(trainSalesColl)
        # TODO: need to handle when trainDeptDoc is None, fall back to Regression for the time being
        if trainDeptDoc == None:
            return self.forecastRegression(trainSalesColl)

        forecastResults =[]
        salesColumn = trainDeptDoc.salesColumn.tolist()

        alpha = 0.2
        for i in range(0, self.recordNumber):
            esForecast = self.calculateES(alpha, salesColumn, numpy.mean(salesColumn))
            forecastResults.append(esForecast)
            salesColumn.append(esForecast)
        return forecastResults

    # ---------- This block is for ANN --------
    def normArrayRange(self, arr, min, max):
        newMin = -1.0
        newMax = 1.0
        oldRange = float((max - min))  
        newRange = float((newMax - newMin))

        normArr = (((arr - min) * newRange) / oldRange) + newMin

        return numpy.round(normArr, decimals = 8)

    def renormArrayRange(self, normArr, min, max):
        oldMin = -1.0
        oldMax = 1.0
        oldRange = float((oldMax - oldMin))  
        newRange = float((max - min))

        renormArr = (((normArr - oldMin) * newRange) / oldRange) + min

        return numpy.round(renormArr, decimals = 8)

    def trainNeuralNetwork(self, trainData, targetData, minSale, maxSale, windowSize):
        # print "trainData", trainData
        # print "targetData", targetData
        # print "minSale", minSale
        # print "maxSale", maxSale
        targetNorm = self.normArrayRange(targetData, minSale, maxSale)
        # print "targetNorm", targetNorm

        dataSize = len(targetNorm)
        targetNorm = targetNorm.reshape(dataSize, 1)

        # print "minSale", minSale, "maxSale", maxSale

        inputSignature = numpy.array(([minSale, maxSale] * windowSize)).reshape(windowSize, 2).tolist()

        # Create network with 2 inputs, 5 neurons in input layer and 1 in output layer
        net = neurolab.net.newff(inputSignature, [5, 1])
        err = net.train(trainData, targetNorm, epochs=500, show=100, goal=0.02)
        
        return net

    def decideANNWindowSize(self, recordNumber):
        if recordNumber/2 > 0:
            return int(recordNumber/2)
        # elif recordNumber/2 > 0:
        #     return int(recordNumber/2)
        else:
            return 0

    def forecastANN(self, trainSalesColl):
        trainDeptDoc = self.fetchTrainDoc(trainSalesColl)
        # pprint(trainDeptDoc.deptSalesInfo)
        if trainDeptDoc == None:
            return [ trainSalesColl.storeDocs[self.storeId].avgStoreSales ] * self.recordNumber

        windowSize = self.decideANNWindowSize(trainDeptDoc.recordNumber)
        # windowSize = 2
        if windowSize == 0:
            return [ trainSalesColl.storeDocs[self.storeId].avgStoreSales ] * self.recordNumber
        # print "windowSize", windowSize

        salesColumn = trainDeptDoc.salesColumn
        minSale = numpy.amin(salesColumn) * 0.5
        maxSale = numpy.amax(salesColumn) * 1.5
        
        # print "salesColumn", salesColumn

        trainList = [] # all training data e.g. [ [], [] ]
        targetList = [] # numberic targets e.g. [ ]
        head = -1

        # Generate training data from train
        while abs(head) + windowSize <= trainDeptDoc.recordNumber:
            tail = head - windowSize
            trainInstance = salesColumn[tail: head]
            trainList.append(trainInstance)
            targetList.append(salesColumn[head])
            head -= 1

        # Construct initial model
        trainList = numpy.array(trainList)
        targetList = numpy.array(targetList)

        network = self.trainNeuralNetwork(trainList, targetList, minSale, maxSale, windowSize)

        # Make forecasting and add new instance into model
        forecastResults =[]
        for i in range (0, self.recordNumber):
            newTrainInstance = salesColumn[-windowSize: ]
            # print "newTrainInstance:", newTrainInstance
            target = network.sim([newTrainInstance]).flatten()
            realTarget = self.renormArrayRange(target, minSale, maxSale)[0]
            forecastResults.append(realTarget)
            salesColumn = numpy.append(salesColumn, realTarget)
            # Feed new instance for the time being
            # trainList = numpy.vstack((trainList, newTrainInstance))
            # targetList = numpy.append(targetList, realTarget)
            # network = self.trainNeuralNetwork(trainList, targetList, minSale, maxSale, windowSize)

        print 'store', self.storeId, 'dept', self.deptId
        print "forecastResults", forecastResults
        return forecastResults


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


    #--------- Multiple Regression START
    # train collection
    trainSalesColl = WalmartSalesColl('train.csv', 'train', False, [])
    # test collection
    testSalesColl = WalmartSalesColl('test.csv', 'test', False, [])

    forecastResults = []
    for deptDoc in testSalesColl.allDeptdocs:
        forecastResults += deptDoc.forecastRegression(trainSalesColl)
        print forecastResults
        break
    
    # pprint(forecastResults)
    # testSalesColl.outputForecastResult(forecastResults, 'finalResults.csv')
    #--------- Multiple Regression END


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
    
     
