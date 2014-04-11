import sys


class FeaturesColl:
    def __init__(self, fileName):
    	self.lines = self.readFileLines(fileName)
        self.storeFeatureDocs = self.buildStoreFeatureDocs(self.lines)
        
    def buildStoreFeatureDocs(self, lines):
        storeId = 1
        storeFeatureDocs = {}
        storeFeatureLines = []  # Will be passed to StoreFeatureDoc
        for line in lines:
            lineInfo = line.split(',', 1)
            #check store change
            if int(lineInfo[0]) !=storeId:
                storeFeatureDoc = StoreFeatureDoc(storeId, storeFeatureLines)
                storeFeatureDocs[storeId] = storeFeatureDoc # Append to global dict, every store will have a doc

                storeId = int(lineInfo[0])
                storeFeatureLines = []  # clear list
            
            storeFeatureLines.append(line)

        # feed last doc
        storeFeatureDoc = StoreFeatureDoc(storeId, storeFeatureLines)
        storeFeatureDocs[storeId] = storeFeatureDoc

        return storeFeatureDocs

    def readFileLines(self, fileName):
        f = open(fileName)
        lines = f.readlines()
        lines.pop(0)
        f.close()
        return lines


class StoreFeatureDoc:
    def __init__(self, storeId, storeFeatureLines):
        self.storeId = storeId
        self.storeFeatureLines = storeFeatureLines
        self.recordNumber = len(storeFeatureLines)
        # featureInstances is a dict looks like {1: {Temp: 20, Fuel:3.66, MD1: 1001}, 2: {}}
        self.featureVectors = self.buildFeatureInstances(storeFeatureLines)
    
    def buildFeatureInstances(self, storeFeatureLines):
        featureLineDict = {} 
        for line in self.storeFeatureLines:
            featureInfo = line.split(',')
            featureLineDict[featureInfo[1]] = {'Temperature': featureInfo[2], \
                                               'Fuel_Price': featureInfo[3], \
                                               'MarkDown1': featureInfo[4], \
                                               'MarkDown2': featureInfo[5], \
                                               'MarkDown3': featureInfo[6], \
                                               'MarkDown4': featureInfo[7], \
                                               'MarkDown5': featureInfo[8], \
                                               'CPI': featureInfo[9], \
                                               'Unemployment': featureInfo[10], \
                                               'IsHoliday': featureInfo[11]}

        return featureLineDict
