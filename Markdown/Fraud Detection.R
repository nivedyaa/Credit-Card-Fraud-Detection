---
title: "Fraud Detection"
author: "Vinod"
date: "July 6, 2019"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
```

#Required Packages
```{r}
#install.packages('keras')
#install.packages('tensorflow')
#install.packages('neuralnet')
#install.packages('randomForest')
library(caret)
library(pROC)
library(mlbench)
library(caretEnsemble)
library(ROSE)
library(smotefamily)
library(tidyverse)
library(here)
library(randomForest)
```

#Import Dataset
```{r}
data <- read_csv(here("Data","creditcardcsvpresent2.csv"))
data$class[data$class==0] <- 'No'
data$class[data$class==1] <- 'Yes'
#data$class <- as.factor(data$class)
#pos <- c(which(data$Is.a.Fraud.merchant.==1))
#data_pos=data[pos,]
#data_pos$isFradulent
#data_neg=data[-pos,]
#data_neg <- data_neg[1:448,]
#data <- rbind(data_pos,data_neg)
str(data)
data <- subset(data,select = -c(1))
#data[,1:9] <- scale(data[,1:9])
#head(data)
#data$class <- factor(data$class)
```



#Accuracy Function
```{r}
c_accuracy=function(actuals,classifications){
  df=data.frame(actuals,classifications);
  
  
  tp=nrow(df[df$classifications==1 & df$actuals==1,]);        
  fp=nrow(df[df$classifications==1 & df$actuals==0,]);
  fn=nrow(df[df$classifications==0 & df$actuals==1,]);
  tn=nrow(df[df$classifications==0 & df$actuals==0,]); 
  
  
  recall=tp/(tp+fn)
  precision=tp/(tp+fp)
  accuracy=(tp+tn)/(tp+fn+fp+tn)
  tpr=recall
  fpr=fp/(fp+tn)
  fmeasure=2*precision*recall/(precision+recall)
  scores=c(recall,precision,accuracy,tpr,fpr,fmeasure,tp,tn,fp,fn)
  names(scores)=c("recall","precision","accuracy","tpr","fpr","fmeasure","tp","tn","fp","fn")
  #print(scores)
  return(scores);
}
```


#Training and Testing
```{r}
set.seed(7)
names(data)[names(data) == 'Average Amount/transaction/day'] <- 'Average_Amount_transaction_day'
names(data)[names(data) == 'Is declined'] <- 'Is_declined'
names(data)[names(data) == 'Total Number of declines/day'] <- 'Total_Number_of_declines_day'
names(data)[names(data) == '6-month_chbk_freq'] <- 'Six_month_chbk_freq'
names(data)[names(data) == '6_month_avg_chbk_amt'] <- 'Six_month_avg_chbk_amt'


ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.75,0.25))
training <- data[ind==1,]
testing <- data[ind==2,]

data_balance <- ovun.sample(class~ ., data = training, method = "both", p=.5, N = nrow(training))$data
training <- data_balance
head(data_balance)
table(training$class)
table(data_balance$class)
training$class <- factor(training$class)

```


#k fold - KNN
```{r}
set.seed(7) 
train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE,  
                              summaryFunction = twoClassSummary, savePredictions = TRUE)
model_knn <- train(class ~., 
                   data = training, 
                   method = "knn",
                   trControl = train.control,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   tuneGrid = expand.grid(k=1:60))
pred_knn<-predict(model_knn,training,class="probs")
confusionMatrix(training$class,pred_knn)
```



#K fold - Neural Network
```{r}
set.seed(7)
train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, verboseIter = TRUE,
                              summaryFunction = twoClassSummary, savePredictions = TRUE)
model_nn <- train(class~., data = training, method = "nnet",metric="ROC",
                  trControl = train.control,preProc = c("center", "scale"))

pred_nn<-predict(object = model_nn,training)
cmnn <- confusionMatrix(training$class,pred_nn)
#broom::tidy(cmnn)
```



#K fold - Logistic Regression

```{r}
set.seed(7)
train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
summaryFunction = twoClassSummary, savePredictions = TRUE)
model_lr <- train(class~., data = training, method = "regLogistic",metric="ROC",
trControl = train.control,preProc = c("center", "scale"))
#model_lr<-train(class ~ Average.Amount.transaction.day + Transaction_amount + 
#                 Is.declined + Total.Number.of.declines.day + isForeignTransaction + 
#                isHighRiskCountry + X6.month_chbk_freq, data = training,
#             method = "regLogistic",metric="ROC",
#            trControl = train.control)
pred_lr<-predict(object = model_lr,training)
confusionMatrix(training$class,pred_lr)
```

#Random Forset
```{r}
set.seed(7)
model_rf <- randomForest(class ~ ., data=training,
                   ntree = 300,
                   mtry = 6,
                   importance = TRUE,
                   proximity = TRUE)
#train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                #              summaryFunction = twoClassSummary, savePredictions = TRUE)
#model_rf <- train(class~., data = training, method = "rf",metric="ROC",
              #    trControl = train.control,preProc = c("center", "scale"))
#pred_rf <-predict(object = model_rf,training)
#confusionMatrix(training$class,pred_rf)
```



#k fold - SVM
```{r}

set.seed(7)
train.control <- trainControl(method = "cv", number = 10,classProbs = TRUE,summaryFunction = twoClassSummary,
                              savePredictions = TRUE)
model_svm <- train(class~., data = training, method = "svmRadial",metric="ROC",
                  trControl = train.control,preProc = c("center", "scale"),
                   tunelength=10,verbose = FALSE)
pred_svm <-predict(object = model_svm,training)
confusionMatrix(training$class,pred_svm)

```

#Model Summary
```{r}
print(model_knn)
print(model_nn)
print(model_lr)
print(model_rf)
print(model_svm)

plot(model_knn)
plot(model_nn)
plot(model_lr)
plot(model_rf)
plot(model_svm)
```



#Performance of Models
```{r}
set.seed(7)
testing$class<-as.factor(testing$class)
pred_rf_prob<-predict(model_rf,testing)
confusionMatrix(testing$class,pred_rf_prob)
pred_knn_prob<-predict(model_knn,testing)
confusionMatrix(testing$class,pred_knn_prob)
pred_lr_prob<-predict(model_lr,testing)
confusionMatrix(testing$class,pred_lr_prob)
pred_nn_prob<-predict(model_nn,testing)
confusionMatrix(testing$class,pred_nn_prob)
pred_svm_prob<-predict(model_svm,testing)
confusionMatrix(testing$class,pred_svm_prob)
```







#Ensemble
```{r}
# testing$pred_majority<- c()
# sum<- c()
# for (i in 1:length(pred_svm_prob)){
# sum[i] = pred_rf_prob[i]+pred_knn_prob[i]+pred_lr_prob[i]+pred_nn_prob[i]+pred_svm_prob[i]
# testing$pred_majority[i]<- (ifelse(sum[i]>=3,1,0))
# }
# #Ensemble Accuracy
# confusionMatrix(testing$class,testing$pred_majority)


set.seed(7)
control <- trainControl(method="cv", number=10, savePredictions= TRUE,
                        index = createFolds(testing$class, 5),classProbs=TRUE, summaryFunction = twoClassSummary)
algorithmList <- c('nnet', 'knn', 'regLogistic', 'rf', 'svmRadial')
models <- caretList(class~., data = testing ,metric="Accuracy", trControl=control, methodList=algorithmList,preProc = c("center", "scale"))
results <- resamples(models)
summary(results)
dotplot(results)
modelCor(results)
splom(results)
# stack using glm
stackControl <- trainControl(method="cv", number=10, savePredictions=TRUE, classProbs=TRUE, summaryFunction = twoClassSummary)
set.seed(7)
stack.glm <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.glm)
acc_ensemble1 <- predict(stack.glm, newdata=testing)
testing$class <- as.factor(testing$class)
acc_ensemble1 <- as.factor(acc_ensemble1)
confusionMatrix(testing$class,acc_ensemble1)
str(acc_ensemble1)
str(testing$class)

```

#Significance of Variables
```{r}
sig <- list(RandomForest=varImp(model_rf),KNN=varImp(model_knn),LogReg=varImp(model_lr),NeuralNetwork=varImp(model_nn),SVM=varImp(model_svm))
sig
```

#further Cleaning
```{r}
testing$class<-as.character(testing$class)
testing$class<-replace(testing$class,c(which(testing$class=='No')),c(0))
testing$class<-replace(testing$class,c(which(testing$class=='Yes')),c(1))
testing$class<-as.numeric(testing$class)

pred_rf_prob <- as.character(pred_rf_prob)
pred_rf_prob <- replace(pred_rf_prob,c(which(pred_rf_prob=='No')),c(0))
pred_rf_prob <- replace(pred_rf_prob,c(which(pred_rf_prob=='Yes')),c(1))
pred_rf_prob <- as.numeric(pred_rf_prob)

pred_knn_prob <- as.character(pred_knn_prob)
pred_knn_prob <- replace(pred_knn_prob,c(which(pred_knn_prob=='No')),c(0))
pred_knn_prob <- replace(pred_knn_prob,c(which(pred_knn_prob=='Yes')),c(1))
pred_knn_prob <- as.numeric(pred_knn_prob)

pred_lr_prob <- as.character(pred_lr_prob)
pred_lr_prob <- replace(pred_lr_prob,c(which(pred_lr_prob=='No')),c(0))
pred_lr_prob <- replace(pred_lr_prob,c(which(pred_lr_prob=='Yes')),c(1))
pred_lr_prob <- as.numeric(pred_lr_prob)

pred_nn_prob <- as.character(pred_nn_prob)
pred_nn_prob <- replace(pred_nn_prob,c(which(pred_nn_prob=='No')),c(0))
pred_nn_prob <- replace(pred_nn_prob,c(which(pred_nn_prob=='Yes')),c(1))
pred_nn_prob <- as.numeric(pred_nn_prob)

pred_svm_prob <- as.character(pred_svm_prob)
pred_svm_prob <- replace(pred_svm_prob,c(which(pred_svm_prob=='No')),c(0))
pred_svm_prob <- replace(pred_svm_prob,c(which(pred_svm_prob=='Yes')),c(1))
pred_svm_prob <- as.numeric(pred_svm_prob)
```


#Accuracy
```{r}
acc_rf<-c_accuracy(testing$class, pred_rf_prob)
acc_knn<-c_accuracy(testing$class, pred_knn_prob)
acc_lr<-c_accuracy(testing$class, pred_lr_prob)
acc_nn<-c_accuracy(testing$class, pred_nn_prob)
acc_svm<-c_accuracy(testing$class, pred_svm_prob)
acc_ensemble<-c_accuracy(testing$class, acc_ensemble1)
accuracy <- matrix(c(acc_rf,acc_knn,acc_lr,acc_nn,acc_svm,acc_ensemble),ncol = 6)
colnames(accuracy) <- c("Random Forest", "K-Nearest Neighbours", "Logistic Regression", "Neural Network","SVM","Ensemble")
rownames(accuracy) <- c("Recall","Precision","Accuracy","TPR","FPR","FMeasure","TP","TN","FP","FN")
write.csv(accuracy,"accuracy.csv")
```



#ROC Curve
```{r}

plot(roc(testing$class,pred_rf_prob), print.auc = TRUE, col = "blue")
plot(roc(testing$class,pred_knn_prob), print.auc = TRUE, col = "red", print.auc.y = .4, add = TRUE)
plot(roc(testing$class,pred_lr_prob), print.auc = TRUE, col = "orange", print.auc.y = .3, add = TRUE)
plot(roc(testing$class,pred_nn_prob), print.auc = TRUE, col = "black", print.auc.y = .2, add = TRUE)
plot(roc(testing$class,pred_svm_prob), print.auc = TRUE, col = "green", print.auc.y = .1, add = TRUE)
legend(0.2,0.4,legend=c("RandomForest", "KNN", "LogReg", "NeuralNetwork", "SVM"),
       col=c("blue", "red","orange","black","green"), 
       lty=1:1, cex=0.8)

pred_rf_prob[1]+pred_knn_prob[1]+pred_lr_prob[1]+pred_nn_prob[1]+pred_svm_prob[1]
```

#THE BIG TABLE
```{r}
big <- matrix(c(pred_rf_prob,pred_knn_prob,pred_lr_prob,pred_nn_prob,pred_svm_prob,acc_ensemble1),ncol = 6)
colnames(big) <- c("Random Forest", "K-Nearest Neighbours", "Logistic Regression", "Neural Network","SVM","Ensemble")
#testing$pred_majority<-as.factor(testing$pred_majority)
#testing$class<-as.factor(testing$class)
write.csv(big,"big.csv")
```





