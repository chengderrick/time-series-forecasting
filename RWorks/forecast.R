#set options to make sure scientific notation is disabled when writing files
WORKING_DIRECTORY = "./"
options(stringsAsFactors = FALSE)
setwd(WORKING_DIRECTORY)
library(timeDate)
options(scipen=500)
#read in data
# dfStore <- read.csv(file='./stores.csv')
# dfSubmission = read.csv(file='./sampleSubmission.csv',header=TRUE,as.is=TRUE)

dfTrain <- read.csv(file='./mock.train')
dfTrain$id = as.character(paste(dfTrain$Store, dfTrain$Dept, sep="_"))
dfTrain$days = as.POSIXlt(x = dfTrain$Date)$yday
dfTrain$weeks = as.integer(dfTrain$days/7+1)
dfTrain$year = as.numeric(substr(dfTrain$Date,1,4))
dfTrain$month = as.numeric(substr(dfTrain$Date,6,7))
dfTrain$day = as.numeric(substr(dfTrain$Date,9,10))

dfTest <- read.csv(file='./mock.test')
dfTest$id = as.character(paste(dfTest$Store, dfTest$Dept, sep="_"))
dfTest$dateId = as.character(paste(dfTest$Store, dfTest$Dept, dfTest$Date, sep="_"))
dfTest$days = as.POSIXlt(x = dfTest$Date)$yday
dfTest$weeks = as.integer(dfTest$days/7+1)
dfTest$year = as.numeric(substr(dfTest$Date,1,4))
dfTest$month = as.numeric(substr(dfTest$Date,6,7))
dfTest$day = as.numeric(substr(dfTest$Date,9,10))
dfTest$Weekly_Sales <- 0

by(dfTest, 1:nrow(dfTest), function(testRow) {
  
  predSales = dfTrain[dfTrain$id == testRow$id & dfTrain$year == (testRow$year-1) & dfTrain$weeks == testRow$weeks,]
  testRow$Weekly_Sales = as.numeric(predSales$Weekly_Sales)
})

dfTest #Doesn't change!
