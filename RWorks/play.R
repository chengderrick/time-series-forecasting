#set options to make sure scientific notation is disabled when writing files
WORKING_DIRECTORY = "~/Time Series/Forecast/RWorks"
#WORKING_DIRECTORY = "./"
options(stringsAsFactors = FALSE)
setwd(WORKING_DIRECTORY)
library(timeDate)
options(scipen=500)
#read in data
# dfStore <- read.csv(file='./stores.csv')


dfTrain <- read.csv(file='./train.csv')
dfTrain$id = as.character(paste(dfTrain$Store, dfTrain$Dept, sep="_"))
dfTrain$days = as.POSIXlt(x = dfTrain$Date)$yday
dfTrain$weeks = as.integer(dfTrain$days/7+1)
dfTrain$year = as.numeric(substr(dfTrain$Date,1,4))
dfTrain$month = as.numeric(substr(dfTrain$Date,6,7))
dfTrain$day = as.numeric(substr(dfTrain$Date,9,10))
dfTrain$fakeId = as.character(paste(dfTrain$Store, dfTrain$Dept, (dfTrain$year), dfTrain$weeks, sep="_"))

dfTest <- read.csv(file='./test.csv')
dfTest$id = as.character(paste(dfTest$Store, dfTest$Dept, sep="_"))
dfTest$dateId = as.character(paste(dfTest$Store, dfTest$Dept, dfTest$Date, sep="_"))
dfTest$days = as.POSIXlt(x = dfTest$Date)$yday
dfTest$weeks = as.integer(dfTest$days/7+1)
dfTest$year = as.numeric(substr(dfTest$Date,1,4))
dfTest$month = as.numeric(substr(dfTest$Date,6,7))
dfTest$day = as.numeric(substr(dfTest$Date,9,10))
dfTest$fakeId = as.character(paste(dfTest$Store, dfTest$Dept, (dfTest$year-1), dfTest$weeks, sep="_"))
dfTest$Weekly_Sales <- 0

m <- merge(x=dfTrain, y=dfTest, by="fakeId", all.y=TRUE)
m$Weekly_Sales.x[is.na(m$Weekly_Sales.x)] <- 1000

print("pass here")
output <- data.frame(m$dateId, m$Weekly_Sales.x, stringsAsFactors=FALSE)
write.table(output,
            file='./outputFinal.csv',
            sep=',', row.names=FALSE, quote=FALSE)
# fuck end

