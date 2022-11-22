#libraries and packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(rminer)) install.packages("rminer", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")


library(tidyverse) #data science package
library(caret) #library classification and regression training
library(rpart) #library for data partition and data preprocessing
library(rpart.plot) #Data visualization for decision trees
library(rminer) #Data mining classification and regression methods
library(nnet) #Neural Network
library(caTools) #moving windows statistics


#Set random seed
set.seed(100)

#Load Customer Data Set

download.file(url="https://raw.githubusercontent.com/manirou-github/machine_learning_project/main/input_data.csv", destfile = paste(getwd(),"/","input_dataset.csv",sep=""), cacheOK=TRUE, mode="wb",method = "libcurl")
input_data <- read_csv(paste(getwd(),"/","input_dataset.csv",sep=""))
input_data %>% column_to_rownames(., var = "custumer_id")

#### DATA EXPLORATION ####

#Description Statistics and Data cleaning
str(input_data) #To gain structure of the data
summary(input_data) #To get an understanding of the data

#Customer churn overview
ggplot(input_data,aes(x=status))+
  geom_histogram(binwidth = 1, fill = c("Blue","Red"))+
  labs(title="Customer telco churn",x="Churn",y="Frequency")

#Customer AON(Age On Network) overview
ggplot(input_data,aes(x=tenure_day/30))+
  geom_histogram(color="Gray", binwidth = 3, fill = "Blue")+
  labs(x="Age On Network of Customer",y="Count",title="Histogram of Age On Network of Customers")
mean(input_data$tenure_day/30) # 45 Months

#Customer Geographic Repartition
ggplot(input_data,aes(x=region))+
  geom_bar(stat="count", fill = "Magenta")+
  labs(x="Region",y="Count",title="Customer Area Repartition")


##### Model Building #####

#data cleaning
region_longer <- input_data %>% 
       mutate(flag=1) %>% 
       replace_na(list(region="unknown")) %>% 
       select(custumer_id,region,flag) %>% 
       pivot_wider(names_from = region, values_from = flag, values_fill = 0)

tmp <- input_data %>% 
       inner_join(region_longer,by=c("custumer_id"="custumer_id")) %>% 
       mutate(tenure_month=tenure_day/30) %>% 
       select(custumer_id,tenure_month,status,balance_amnt,rev_o_tot_amnt,
              rev_voix_pyg_amnt,rev_sms_pyg_amnt,rev_data_pyg_amnt,
              rev_mms_pyg_amnt,rev_subs_amnt,NORTH,EAST,WEST,SOUTH,`CAPITAL CITY`,`BUSINESS CITY`,`NORTH WEST`,`SOUTH EAST` )
       
  
input_data <- tmp

rm(tmp,region_longer)


#Data partition 80/20 split

input_data$status <- factor(input_data$status)

split_set <- createDataPartition(y=input_data$status, p=.80, list = FALSE)
train_set <- input_data[split_set,]
test_set <- input_data[-split_set,]

#Model 1 : Logistic Regression
lr_model <- glm(status~tenure_month+balance_amnt+rev_o_tot_amnt+rev_voix_pyg_amnt+
                  rev_sms_pyg_amnt+rev_data_pyg_amnt+rev_mms_pyg_amnt+rev_subs_amnt
                    +NORTH+EAST+WEST+SOUTH+`CAPITAL CITY`+`BUSINESS CITY`+`NORTH WEST`+`SOUTH EAST`, family = "binomial", train_set)
summary(lr_model)


#Model prediction
lr_prediction <- predict(lr_model, test_set, type = "response")

#Generate ROC curve
model_AUC <- colAUC(lr_prediction,test_set$status,plotROC = T)
abline(h=model_AUC, col = "Blue")
text(.2,.9,cex = .8, labels = paste("Optimal Cutoff", round(model_AUC,4)))

#Convert probabilities to class
churn_class <- ifelse(lr_prediction>0.76,1,0)

churn_class <- factor(churn_class)

#Confusion Matrix
confusionMatrix(churn_class,test_set$status)


#Model 2 : Neural Network

nn_model <- multinom(status~tenure_month+balance_amnt+rev_o_tot_amnt+rev_voix_pyg_amnt+
                       rev_sms_pyg_amnt+rev_data_pyg_amnt+rev_mms_pyg_amnt+rev_subs_amnt
                     +NORTH+EAST+WEST+SOUTH+`CAPITAL CITY`+`BUSINESS CITY`+`NORTH WEST`+`SOUTH EAST`, data = train_set) #to assign logistic regression based neural network to a name
summary(nn_model) #summary of the model

#Model prediction
nn_prediction <- predict(nn_model,test_set) #Prediction using Neural model in conjunction with the testing set
prediction_table <- table(nn_prediction, test_set$status) #Put information into confusion matrix
prediction_table #print confusion matrix

#correct classification
sum(diag(prediction_table))/sum(prediction_table)

#Missclassification Rate
1-sum(diag(prediction_table))/sum(prediction_table)


###Model 3 : Decision Trees ###

#Data partition

split_set <- sample(2,nrow(input_data),replace=TRUE,prob = c(0.80,0.20))

train_set <- input_data[split_set==1,]

test_set <- input_data[split_set==2,]

dtree_model <- rpart(status~tenure_month+balance_amnt+rev_o_tot_amnt+rev_voix_pyg_amnt+
                       rev_sms_pyg_amnt+rev_data_pyg_amnt+rev_mms_pyg_amnt+rev_subs_amnt
                     +NORTH+EAST+WEST+SOUTH+`CAPITAL CITY`+`BUSINESS CITY`+`NORTH WEST`+`SOUTH EAST`,data = train_set)
summary(dtree_model)


#To plot the decision tree
rpart.plot(dtree_model)

#churn prediction
dtree_prediction <- predict(dtree_model,train_set,type="class")

#confusion matrix
confusionMatrix(dtree_prediction,train_set$status)


### test_set 

dtree_model <- rpart(status~.,data = test_set)
summary(dtree_model)

#To plot the decision tree
rpart.plot(dtree_model)

#churn prediction
dtree_prediction <- predict(dtree_model,test_set,type="class")


#Missclassification Rate
confusionMatrix(dtree_prediction,test_set$status)




