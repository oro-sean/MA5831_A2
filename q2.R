## set up environment and import data file

rm(list = ls()) # removes all variables
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014") # clear console


# import required packages 
library(DataExplorer, quietly = TRUE)
library(readxl, quietly = TRUE)
library(ggplot, quietly = TRUE)
library(caret, quietly = TRUE)
library(doParallel,quietly = TRUE)
library(parallel, quietly = TRUE)
library(xgboost, quietly = TRUE)

file <- "CreditCard_Data.xls" # set source data file name
rawData <-  read_excel("CreditCard_Data.xls", 
                                         col_types = c("numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric", "numeric", "numeric", 
                                                       "numeric"), skip = 1) # the data is imported, the .csv file has headers which will be used as the column names of the data frame, any stirngs in the data set will be treated as factors

rawData <- data.frame(rawData) # change data structure to dataframe
makeFact <- c(3,4,5,7,8,9,10,11,12,25) # column index's of categorical variables
for (i in makeFact){ # loop through all categorical variables and make factors
  rawData[ , i] <- as.factor(rawData[ , i])
}
colnames(rawData)[25] <- "default" # make column name shorter

## data exploration and visualization
str(rawData) # returns the structure of the data.frame 
introduce(rawData) # returns some basic information about the data
head(rawData) # shows the first 6 lines of data
summary(rawData) # returns a basic statistical summary of the data
#plot_histogram(rawData)
#plot_bar(rawData)
#plot_qq(rawData)

## Pass rawData to modelData
modelData <- rawData

## Test Training Split
set.seed(123)
train_index <- createDataPartition(modelData$default, p=0.7, list = FALSE) # returns numerical vector of the index of the observations to be included in the training set.
predictors <- names(modelData[ , 2:24]) # return vector of column names, removing "default" as it is the response variable and ID as it is the observation ID

testData <- modelData[-train_index, ] # create data.frame of test data
trainingPredictors <- data.matrix(modelData[train_index,predictors]) # create data.frame of training predictors
trainingResponse <- as.factor(modelData[train_index, "default"]) # create vector of training responses

plot_bar(testData$default, title = "Response Frequency Test Data")
plot_bar(trainingResponse, title = "Response Frequency Training Data")

rows_trainingPred <- nrow(trainingPredictors) # return number of training observations
columns_trainingPred <-ncol(trainingPredictors) # return number of training predictors
rows_trainingResponse <- length(trainingResponse) # return length of training response

rm(rawData, file, i, makeFact) # remove unused variables

## Fit Boosted Tree

(modelLookup("xgbTree")) # return available tuning parameters
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)  # set train control variable to specify 10 fold cross validation

tuneGrid <- expand.grid(
  nrounds = c(1,50,100),
  eta = c(1),
  max_depth = c(5),
  gamma = c(.01),
  colsample_bytree = 1,
  min_child_weight = c(1),
  subsample = 1)




xgb <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid,
  method = "xgbTree"
)

xgb$results


## Fit SVM

## Center and scale numeric predictors
set.seed(123)
train_index_01 <- createDataPartition(modelData$default, p=0.01, list = FALSE)

testData <- modelData[-train_index_01, ] # create data.frame of test data
trainingPredictors <- modelData[train_index_01,predictors] # create data.frame of training predictors
trainingResponse <- as.factor(modelData[train_index_01, "default"]) # create vector of training responses

preProc_train <- preProcess(trainingPredictors, method = c("center", "scale"))

trainingPredictors <- data.matrix(predict(preProc_train, newdata = trainingPredictors))

(modelLookup("svmRadial")) # return available tuning parameters

train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)  # set train control variable to specify 10 fold cross validation

cl <- makeCluster(10)
registerDoParallel(cl)
startTime <- Sys.time()

tuneGrid_lin <- expand.grid(
  C = c(.001,1,10)
  )

svm_lin <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_lin,
  method = "svmLinear"
)

tuneGrid_pol <- expand.grid(
  degree = c(1,5,10),
  scale = 1,
  C = c(.001,1,10)
)

svm_pol <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_pol,
  method = "svmPoly"
)

tuneGrid_rad <- expand.grid(
  sigma = 1:10,
  C = c(.001,1,10)
)

svm_rad <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_rad,
  method = "svmRadial"
)

stopCluster(cl)
finishTime <- Sys.time()
(finishTime - startTime)

max(svm_lin$results$Kappa)
max(svm_pol$results$Kappa)
max(svm_rad$results$Kappa)

svm_pol$results

tuneGrid_pol <- expand.grid(
  degree = c(1,5,10),
  scale = c(10,20,30),
  C = c(.0001,.01,1,100)
)

svm_pol <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_pol,
  method = "svmPoly"
)

## train SVM
set.seed(123)

testData <- modelData[-train_index, ] # create data.frame of test data
trainingPredictors <- modelData[train_index,predictors] # create data.frame of training predictors
trainingResponse <- as.factor(modelData[train_index, "default"]) # create vector of training responses

preProc_train <- preProcess(trainingPredictors, method = c("center", "scale"))

trainingPredictors <- data.matrix(predict(preProc_train, newdata = trainingPredictors))

train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)  # set train control variable to specify 10 fold cross validation

cl <- makeCluster(10)
registerDoParallel(cl)
startTime <- Sys.time()


tuneGrid_pol <- expand.grid(
  degree = c(1,10,100),
  scale = c(1),
  C = c(1)
)

svm_pol <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_pol,
  method = "svmPoly"
)

stopCluster(cl)
finishTime <- Sys.time()
(finishTime - startTime)
svm_pol$results$Kappa

