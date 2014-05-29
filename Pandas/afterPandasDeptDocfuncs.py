      
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
   