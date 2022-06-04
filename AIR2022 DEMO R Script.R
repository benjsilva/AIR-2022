#--------------------------------------------------------------#
# Machine-Learning Models for Term Success & Timely Graduation #
# AIR FORUM 2022                                               #
# Benjamin Silva                                               #
# Florida Atlantic University                                  #
# silvab@fau.edu                                               #
#--------------------------------------------------------------#

library(readr)

dataset <- read_csv(url("https://raw.githubusercontent.com/benjsilva/AIR-2022/main/AIR_RSTUDIO_DEMO.csv"))

# AIR 2022 R example

## Load caret package

library(caret)

## Set factors

factor_names <- c( "FEMALE_IND"	       
                   ,"INSTATE_IND"   
                   ,"ADMIT_SEMESTER"
                   ,"LIVES_ON_CAMPUS_IND"
                   ,"FULLTIME_IND"    
                   ,"ARTS_IND"
                   ,"BUSINESS_IND"
                   ,"EDUCATION_IND"
                   ,"ENGINEERING_IND"
                   ,"HONORS_IND"
                   ,"NURSING_IND"
                   ,"SCIENCE_IND"
                   ,"UNDECIDED_IND"
                   ,"PELL_IND"
                   ,"STEM_IND"
                   ,"EARNED_AA_IND"
                   ,"TIMELY_GRAD_IND"
                   ,"TERM_SUCCESS_IND")

dataset[factor_names] <- lapply(dataset[factor_names], factor)

dataset$TIMELY_GRAD_DESC <- ifelse(dataset$TIMELY_GRAD_IND == 1,'yes','no')
dataset$TIMELY_GRAD_DESC <- as.factor(dataset$TIMELY_GRAD_DESC)

## Subset Training & Application data sets

training_data <- subset(dataset, DATA_TYPE == 'Training')
application_data <- subset(dataset, DATA_TYPE == 'Application')

## Subset data sets by semester (Fall), student type, cohort and years enrolled

### 4th year FTICs

FTIC_4YR_TRAIN <- as.data.frame(subset(training_data, SEMESTER == 'Fall' & STU_TYPE_DESC == 'FTIC' 
                         & COHORT_YEAR == 4))


FTIC_4YR_APPLY <- as.data.frame(subset(application_data, SEMESTER == 'Fall' & STU_TYPE_DESC == 'FTIC' 
                         & COHORT_YEAR == 4))


## Set variable list for training model (do not include outcome variable
## or unwanted predictors)

`%!in%` <- Negate(`%in%`)

F4_predictors <- names(application_data)[
  names(application_data) %!in% c('GENERIC_ID', 'RANDOM_NAME'
                                  , 'SEMESTER', 'FTIC_IND', 'STU_TYPE_DESC'
                                  ,'COHORT_YEAR', 'TRANSFER_GPA'
                                  ,'CLASS_LEVEL_DESC', 'COLLEGE_DESC', 'GENDER_DESC' 
                                  ,'FIRST_TERM_IND', 'EARNED_AA_IND'
                                  ,'UNDECIDED_IND'
                                  ,'ARTS_IND', 'DATA_TYPE'
                                  ,'TIMELY_GRAD_IND', 'TIMELY_GRAD_DESC'
                                  ,'LIVES_ON_CAMPUS_IND', 'COMPLETION_RATE'
                                  ,'AVG_WD', 'CRS_DIFF_CNT', 'TERM_SUCCESS_IND'  
                                  )]

## Train model

set.seed(1234)

### 3 fold cross validation

gradControl <- trainControl(method='cv', number=3, returnResamp='none'
                            , summaryFunction = twoClassSummary, classProbs = TRUE)

### GBM implementation

F4_Model <- train(FTIC_4YR_TRAIN[,F4_predictors], 
                  FTIC_4YR_TRAIN$TIMELY_GRAD_DESC, 
                  method='gbm', 
                  trControl=gradControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))


## Variable Importance

summary(F4_Model)


## Predicted probabilities

F4_probs <- predict(object=F4_Model, FTIC_4YR_APPLY[,F4_predictors],
                    type='prob')

head(F4_probs)

## Combine predictions with test dataframe

FTIC_4YR_APPLY$TIMELY_GRAD_PROB <- F4_probs[,2]

## Plot Accuracy & ROC Curve

library(ROCR)

#Timely grad accuracy over probability cutoff
pred_acc = prediction(FTIC_4YR_APPLY$TIMELY_GRAD_PROB, FTIC_4YR_APPLY$TIMELY_GRAD_IND)
perf = performance(pred_acc, "acc")

# Plot Accuracy over probability cutoff

plot(perf) 

#Timely grad ROC Curve
pred_rocr = prediction(FTIC_4YR_APPLY$TIMELY_GRAD_PROB, FTIC_4YR_APPLY$TIMELY_GRAD_IND)
roc = performance(pred_rocr, "tpr","fpr")


# Plot ROC Curve

plot(roc, colorize = T, lwd = 2)
abline(a = 0, b = 1)

# Get AUC
auc_ROCR <- performance(pred_rocr, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]

auc_ROCR