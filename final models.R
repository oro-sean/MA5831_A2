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

## Pass rawData to modelData
modelData <- rawData

## Test Training Split
set.seed(123)

train_index <- createDataPartition(modelData$default, p=0.7, list = FALSE) # returns numerical vector of the index of the observations to be included in the training set.
predictors <- names(modelData[ , 2:24]) # return vector of column names, removing "default" as it is the response variable and ID as it is the observation ID

testData <- modelData[-train_index, ] # create data.frame of test data
trainingPredictors <- data.matrix(modelData[train_index,predictors]) # create data.frame of training predictors
trainingResponse <- as.factor(modelData[train_index, "default"]) # create vector of training responses

rm(rawData, file, i, makeFact) # remove unused variables

## Fit Boosted Tree

train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)  # set train control variable to specify 10 fold cross validation, allow parrallel for speed, --> not repeatable

tuneGrid_tree <- expand.grid( # set up tuning grid for boosted tree
  nrounds = seq(1,100,10),
  eta = 1,
  max_depth = 1:10,
  gamma = seq(0,1,.01),
  colsample_bytree = 1,
  min_child_weight = 1:10,
  subsample = 1)



XGB <- train( # train boosted tree
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_tree,
  method = "xgbTree"
)



## Fit SVM
set.seed(123)
## Recreat test and training data in data frame with factors to make centrering and scaling easy
testData <- modelData[-train_index, ] # create data.frame of test data
trainingPredictors <- modelData[train_index,predictors] # create data.frame of training predictors
trainingResponse <- as.factor(modelData[train_index, "default"]) # create vector of training responses

preProc_train <- preProcess(trainingPredictors, method = c("center", "scale")) # set up preproc paramateres to centre and scale all numerical factors

trainingPredictors <- data.matrix(predict(preProc_train, newdata = trainingPredictors)) # centre and scale training data

tuneGrid_svm <- expand.grid(
  degree = 1:10,
  scale = c(10,20,30,40,50),
  C = c(1)
)

svm_final <- train(
  x = trainingPredictors,
  y = trainingResponse,
  trControl = train_control,
  tuneGrid = tuneGrid_svm,
  method = "svmPoly"
)

stopCluster(cl)
finishTime <- Sys.time()
(finishTime - startTime)


