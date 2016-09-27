---
title: "Machine Learning Project Report"
author: "Ravi Kumar"
date: "September 26, 2016"
output: html_document
---
#Executive Summary
One thing that people regularly do is to quantify how much of a particular activity they do using modern devices such as  Jawbone Up, Nike FuelBand, Fitbit etc. but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which they have performed the excercise.Also, this report describes how model has been buit, which cross validation has been used, what expected out of sample error is for the model, and why theses choices were made. Also, prediction model developed has been used to predict 20 different test cases. I would like to thank http://groupware.les.inf.puc-rio.br/har for providing data for the project.

###Libraries
Following library has been used throughout the preprocessing, model building and predicting the test data set.

```r
library(caret)
library(kernlab)
library(randomForest)
library(e1071)
```

###Getting the Data and Preprocessing

Getting the raw data and converting raw data into tidy data following a particular processing script are very important steps of modeling before doing any further analysis. Same processing script should be followed to convert training and test raw dataset to tidy dataset which is ready for model development and testing the model.   

####Getting data from Web

```r
#Follwing are the link for training and testing data:
train_Url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_Url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#First new folder named "Machine Learning" has been created (if not already exist) using following code:
if (!file.exists("Machine Learning")){
  dir.create("Machine Learning")
}
#Files are downloaded into "Machine Learning" folder using following command:
download.file(train_Url, destfile = "./Machine Learning/train_raw.csv")
download.file(test_Url, destfile = "./Machine Learning/test_raw.csv")
```
####Loading file on R
File has been read using read.csv command automatically set sep = "," and header = TRUE as data is in .csv format. Also, na.strings has been used to assign NA for blank, #DIV/0! and NA while reading the file.


```r
# Load data into R
train_raw <- read.csv("./Machine Learning/train_raw.csv",na.strings = c("#DIV/0!", "NA"))
test_raw <- read.csv("./Machine Learning/test_raw.csv",na.strings = c("#DIV/0!", "NA"))
```
####Cleaning the data
As there are 19622 observations and multiple columns have NA values. Therefore we can eliminate columns with NA values. 

```r
# remove columns containing more than 95% of NA
train_Without_NA <- train_raw[, colSums(!is.na(train_raw))/nrow(train_raw)>0.95]
test_Without_NA <- test_raw[, colSums(!is.na(test_raw))/nrow(test_raw)>0.95]
## remove columns with zero variance
nsv<- nearZeroVar(train_Without_NA, saveMetrics = TRUE)
train1<-train_Without_NA[, !nsv$nzv]
nsv1<- nearZeroVar(test_Without_NA, saveMetrics = TRUE)
test1<-test_Without_NA[, !nsv1$nzv]
#Remove Identifiers
train_clean <- train1[, 8:ncol(train1)]
test_clean <- test1[, 8:ncol(test1)]
#dummies <- dummyVars(classe~user_name, train_clean)
#z<-predict(dummies, train_clean)
```
###Model Development
Before any model development testing The train data set has been devided in training and testing data set and test data is named as validation dataset. As we can see both training and testing data has same proportion of each category of excercise. 

```r
set.seed(123)
#Check % of each excercise category in train_clean
table(train_clean$classe)/nrow(train_clean)
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```

```r
#dividing data into training, testing and validation data set
inTrain <- createDataPartition (y=train_clean$classe, p=0.7, list = FALSE)
training<- train_clean[inTrain,]
testing<- train_clean[-inTrain,]
dim(training)
```

```
## [1] 13737    52
```

```r
dim(testing)
```

```
## [1] 5885   52
```

```r
#Check % of each excercise category in training
table(training$classe)/nrow(training)
```

```
## 
##         A         B         C         D         E 
## 0.2843416 0.1934920 0.1744195 0.1639368 0.1838101
```

```r
#Check % of each excercise category in testing
table(testing$classe)/nrow(testing)
```

```
## 
##         A         B         C         D         E 
## 0.2844520 0.1935429 0.1743415 0.1638063 0.1838573
```

```r
validation<- test_clean
```
Random forest method has been chosen developing the model as it involves both begging and classification and hence gives most accurate result. If target or respose variable is a factor then classification random forest will be used. 

```r
class(training$classe)
```

```
## [1] "factor"
```
Since class of target is factor, So classification Random Forest will be built. 
#### Building Random Forest using R

```r
# Build random forest model with ntree=500 and variable importance to be assessed as true
modFit <- randomForest(classe ~., training, ntree = 500, importance = T)
plot(modFit)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png)
There is no significant error reduction after 200 decision trees. To find out important variable is also an important step in developing a model. Following plot shows 10 most important variable and also list of variablw with decreasing importance has been created.

```r
varImpPlot(modFit, sort=T,main ="Variable Importance",type=2, n.var=10)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png)

```r
var.imp <- data.frame(importance(modFit,type=2))
# make row names as columns
var.imp$Variables <- row.names(var.imp)
var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),]
```

```
##                      MeanDecreaseGini            Variables
## yaw_belt                    717.96088             yaw_belt
## magnet_dumbbell_z           569.26796    magnet_dumbbell_z
## pitch_forearm               546.13153        pitch_forearm
## pitch_belt                  526.36164           pitch_belt
## magnet_dumbbell_y           464.06117    magnet_dumbbell_y
## roll_forearm                409.96317         roll_forearm
## accel_belt_z                353.48856         accel_belt_z
## magnet_dumbbell_x           349.85191    magnet_dumbbell_x
## magnet_belt_y               331.85052        magnet_belt_y
## magnet_belt_z               330.64675        magnet_belt_z
## roll_dumbbell               320.58726        roll_dumbbell
## accel_dumbbell_y            290.00625     accel_dumbbell_y
## gyros_belt_z                275.78055         gyros_belt_z
## accel_dumbbell_z            250.22799     accel_dumbbell_z
## roll_arm                    245.42078             roll_arm
## accel_forearm_x             230.36237      accel_forearm_x
## magnet_forearm_z            210.40287     magnet_forearm_z
## total_accel_dumbbell        199.78861 total_accel_dumbbell
## magnet_belt_x               196.68370        magnet_belt_x
## accel_arm_x                 195.49228          accel_arm_x
## yaw_dumbbell                190.06593         yaw_dumbbell
## magnet_arm_y                189.23323         magnet_arm_y
## accel_forearm_z             186.59929      accel_forearm_z
## gyros_dumbbell_y            186.35083     gyros_dumbbell_y
## magnet_arm_x                180.43058         magnet_arm_x
## total_accel_belt            179.68207     total_accel_belt
## accel_dumbbell_x            179.05839     accel_dumbbell_x
## yaw_arm                     172.10642              yaw_arm
## magnet_forearm_x            169.89995     magnet_forearm_x
## magnet_forearm_y            160.75286     magnet_forearm_y
## magnet_arm_z                154.21835         magnet_arm_z
## pitch_arm                   136.13120            pitch_arm
## pitch_dumbbell              125.93521       pitch_dumbbell
## accel_arm_y                 120.15292          accel_arm_y
## yaw_forearm                 116.39898          yaw_forearm
## accel_belt_y                111.81637         accel_belt_y
## accel_forearm_y             108.43934      accel_forearm_y
## accel_arm_z                 106.25538          accel_arm_z
## gyros_arm_y                 103.37055          gyros_arm_y
## gyros_dumbbell_x            101.30155     gyros_dumbbell_x
## gyros_forearm_y              99.79020      gyros_forearm_y
## gyros_arm_x                  99.40391          gyros_arm_x
## gyros_belt_y                 95.95699         gyros_belt_y
## accel_belt_x                 88.30824         accel_belt_x
## gyros_belt_x                 84.32129         gyros_belt_x
## total_accel_forearm          84.20492  total_accel_forearm
## total_accel_arm              77.82834      total_accel_arm
## gyros_dumbbell_z             66.32079     gyros_dumbbell_z
## gyros_forearm_z              65.61716      gyros_forearm_z
## gyros_forearm_x              61.33987      gyros_forearm_x
## gyros_arm_z                  44.68522          gyros_arm_z
```
Based on this variable importance table, variables should be selected for any other predtive models.

####In Sample Error
To calculate In Sample error, model developed has been used to predict traning data.

```r
training$predicted <- predict(modFit, training)
confusionMatrix(data= training$predicted, reference = training$classe, positive = 'Yes')
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

The result shows that model is having OBB error of 0%.

###Cross Validation
Cross validation was done using the testing data which was kept aside from our train data. The result shows model is able to predict the testing data perfectly.

```r
testing$predicted <- predict(modFit,testing)
#print(pred1)
confusionMatrix(data = testing$predicted, reference= testing$classe, positive = 'Yes')
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    6    0    0    0
##          B    1 1133   12    0    0
##          C    0    0 1014   14    0
##          D    0    0    0  949    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9947   0.9883   0.9844   1.0000
## Specificity            0.9986   0.9973   0.9971   1.0000   0.9998
## Pos Pred Value         0.9964   0.9887   0.9864   1.0000   0.9991
## Neg Pred Value         0.9998   0.9987   0.9975   0.9970   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1925   0.1723   0.1613   0.1839
## Detection Prevalence   0.2853   0.1947   0.1747   0.1613   0.1840
## Balanced Accuracy      0.9990   0.9960   0.9927   0.9922   0.9999
```
As we can see model has predicted testing data with the accuracy of 99.42% i.e. out of Sample error is 0.58% only.

###Prediction
Now validation data has been predicted using the model.

```r
# to makes levels of factors of validation1 equal to levels in training data set 
levels(validation$new_window)<-levels(training$new_window)
levels(validation$cvtd_timestamp)<-levels(training$cvtd_timestamp)
set.seed(123)
validation$pred <- predict(modFit,validation)
Problem_id <-c(1:20)
Prediction<-validation$pred
result<-data.frame(Problem_id,Prediction)
```
The final result for test data is:

```r
print(result)
```

```
##    Problem_id Prediction
## 1           1          B
## 2           2          A
## 3           3          B
## 4           4          A
## 5           5          A
## 6           6          E
## 7           7          D
## 8           8          B
## 9           9          A
## 10         10          A
## 11         11          B
## 12         12          C
## 13         13          B
## 14         14          A
## 15         15          E
## 16         16          E
## 17         17          A
## 18         18          B
## 19         19          B
## 20         20          B
```
###Conclusion
Random forest is the most suitable method for the given data set but before that tidy data needs to be prepared from raw data using multiple data cleaning and preprocessing processes. Also, both test and training data should have same preprocessing script.   


