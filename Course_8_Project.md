### **Background about the Data**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways.

#### **Data Preparation**

    #Set your working directory
    setwd("C:/Users/dcai0559/Desktop/Coursera/Practical Machine Learning")

    #Load the required packages in this analysis
    library(rpart)          #For Partitioning the data, classification by decision trees, and for regression trees
    library(caret)          #For linear discriminant Analysis

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(randomForest)   #For Creating Random forest for Regression

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(e1071)          #For Linear discriminant Analysis

    #Download the datasets

    url_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    data_training <- read.csv(url(url_training), na.strings=c("NA","#DIV/0!",""))
    data_test <- read.csv(url(url_test), na.strings=c("NA","#DIV/0!",""))

Explore the data.

    names(data_training)
    str(data_training)
    summary(data_training)

Exclude variables.

    #Exclude the first 7 variables as they are just ID's (identifiers)
    data_training2 <- data_training[,-c(1:7)]

    #Exclude the variables that only contains NA's.
    data_training2 <- data_training2[,colSums(is.na(data_training2)) == 0]

    dim(data_training2)

    ## [1] 19622    53

From the original 160 variables, we filtered it out, and now, we're up
with only 53 variables. We can start working on those variables.

#### **Partition the Data**

    part <- createDataPartition(y = data_training2$classe, p=0.75, list=FALSE )
    development <- data_training2[part,]   #This will be used for the model development
    validation <- data_training2[-part,]   #This will be used for testing the model produced

#### **Model Development**

For predictive modelling, we will use several methods: linear
discriminant analysis, Classification Tree model, and Random Forest
model.

    #Linear Discriminant Analysis
    model1 <- train(classe ~ ., data=development, method="lda",na.action = na.exclude)
    #Classification Tree Model
    model2 <- train(classe ~., data=development, method="rpart",na.action = na.exclude)
    #Random Forest model
    model3 <- randomForest(classe ~.,data=development)

#### **Model Assessment**

From the models that we generated using three different methods, we will
assess which of these models is the best to predict.

    #PREDICTIVE MODELLING using the Validation data
    predict1 <- predict(model1,validation)
    predict2 <- predict(model2,validation)
    predict3 <- predict(model3,validation)

    confusionMatrix(predict1, validation$classe)  

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1143  145   94   54   32
    ##          B   31  590   74   43  162
    ##          C  110  132  555   93   87
    ##          D  103   36  113  580   73
    ##          E    8   46   19   34  547
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6964          
    ##                  95% CI : (0.6833, 0.7092)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6156          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8194   0.6217   0.6491   0.7214   0.6071
    ## Specificity            0.9074   0.9216   0.8958   0.9207   0.9733
    ## Pos Pred Value         0.7786   0.6556   0.5681   0.6409   0.8364
    ## Neg Pred Value         0.9267   0.9103   0.9236   0.9440   0.9167
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2331   0.1203   0.1132   0.1183   0.1115
    ## Detection Prevalence   0.2993   0.1835   0.1992   0.1845   0.1334
    ## Balanced Accuracy      0.8634   0.7717   0.7724   0.8211   0.7902

    confusionMatrix(predict2, validation$classe)  

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1291  397  385  361  124
    ##          B   15  316   34  145  117
    ##          C   86  236  436  298  240
    ##          D    0    0    0    0    0
    ##          E    3    0    0    0  420
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5022          
    ##                  95% CI : (0.4881, 0.5163)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3493          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9254  0.33298  0.50994   0.0000  0.46615
    ## Specificity            0.6389  0.92137  0.78760   1.0000  0.99925
    ## Pos Pred Value         0.5047  0.50399  0.33642      NaN  0.99291
    ## Neg Pred Value         0.9557  0.85200  0.88387   0.8361  0.89266
    ## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
    ## Detection Rate         0.2633  0.06444  0.08891   0.0000  0.08564
    ## Detection Prevalence   0.5216  0.12785  0.26427   0.0000  0.08626
    ## Balanced Accuracy      0.7822  0.62717  0.64877   0.5000  0.73270

    confusionMatrix(predict3, validation$classe)  

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    5    0    0    0
    ##          B    1  943    3    0    0
    ##          C    0    1  851    6    1
    ##          D    0    0    1  797    0
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9961         
    ##                  95% CI : (0.994, 0.9977)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9951         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9937   0.9953   0.9913   0.9989
    ## Specificity            0.9986   0.9990   0.9980   0.9998   0.9998
    ## Pos Pred Value         0.9964   0.9958   0.9907   0.9987   0.9989
    ## Neg Pred Value         0.9997   0.9985   0.9990   0.9983   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2843   0.1923   0.1735   0.1625   0.1835
    ## Detection Prevalence   0.2853   0.1931   0.1752   0.1627   0.1837
    ## Balanced Accuracy      0.9989   0.9963   0.9967   0.9955   0.9993

#### **Conclusion**

Based from the three methods, I finally decided to use Random Forest
model as my predictive model as it has the HIGHEST ACCURACY, i.e. 0.9965
(vs. LDA: 0.7015, Classification Tree: 0.5006). Let us now use this to
predict the results using **data\_test**.

    answer <- predict(model3,data_test)

    answer

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
