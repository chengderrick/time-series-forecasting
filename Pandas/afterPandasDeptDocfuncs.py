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