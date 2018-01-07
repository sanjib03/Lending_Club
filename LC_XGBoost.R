
#Program header-----
#
# Description: This program predicts the Good or Bad loan for Lending club loan data.
#
# Input : loan.csv


#load libraries
library(data.table)
#install.packages("RSQLite")
library(RSQLite)
#install.packages("mice")
library(mice)
#install.packages("dummies")
library(dummies)
#install.packages('xgboost')
library(caret)
library(xgboost)
library(randomForest)
install.packages("ggcorrplot")
library(ggcorrplot)



#Load data
# con <- dbConnect(drv=RSQLite::SQLite(), dbname="C:\\Kaggle\\Lending_Club\\database.sqlite")
# table <- dbListTables(con)
# 
# loan_dt1 <- as.data.table(dbGetQuery(con, 'select * from loan'))


#Load data--------
loan_dt<- as.data.table(read.csv('C:/Kaggle/Lending_Club/loan.csv', 
                                  header = TRUE, sep = ',',stringsAsFactors = FALSE))
# check dimenstion
dim(loan_dt)
# See top 50 rows
#View(head(loan_dt,50))


# Create Target variable
bad_indicators <- c("Charged Off ",
                    "Charged Off",
                    "Default",
                    "Does not meet the credit policy. Status:Charged Off",
                    "In Grace Period", 
                    "Late (16-30 days)",
                    "Late (31-120 days)")

loan_dt[, target := as.factor(ifelse(loan_status %in% bad_indicators,1,0))]
loan_dt$loan_status <- NULL

# Intial Checks---------
summary(loan_dt)
str(loan_dt)

# Check the ratio of good to bad loan in the data
loan_dt[, .(count=.N/nrow(loan_dt)), by = target]
# data is imbalanced as we have 92% good loans

# Find the numeric columns
num_col <- sapply(loan_dt, is.numeric) 
num_col <- num_col[num_col]
length(num_col)
# so there are 52 numeric and 23 character column



# Data Cleaning-------
# Remove some features such as id, title, LC page url, etc...
#   The variable policy code has a single value 1 for all rows.
#   The payment date variable do not give any information about credit worthiness.
#   For simplicuty, I am not considering initial status and join status.

drop_col <- c('id','member_id','emp_title','home_ownership','issue_d',
              'url','desc','title', 'zip_code','last_pymnt_d',
              'next_pymnt_d','initial_list_status','last_credit_pull_d'
              ,'policy_code', 'verification_status_joint' )
loan_dt[,(drop_col):=NULL]


## Data Quality check
# Find the data completeness for all columns
data_cmplt_smry <- data.table(var_name=character(),cmplt_pc=numeric())

for(ii in names(loan_dt)){
  temp = data.frame(var_name=ii,
                    cmplt_pc=(1 - round(sum(is.na(loan_dt[[ii]]))/nrow(loan_dt),5))*100
                    )
  data_cmplt_smry = rbind(data_cmplt_smry, temp)
}

# Subset the data having only the variables with more than 40% data completeness
select_col<- as.character(data_cmplt_smry[cmplt_pc>=40]$var_name)
loan_dt_sub <- loan_dt[, select_col, with=FALSE]



# Now we have 12 fields with data completion >40% and <100%

# Out of these, acc_now_delinq, collections_12_mths_ex_med are highly skewed 
# with more than 98% value as 0. So imputing will not improve its variability.
# Hence, removing them.
summary(loan_dt_sub$acc_now_delinq)
nrow(loan_dt_sub[acc_now_delinq==0])/nrow(loan_dt_sub)
loan_dt_sub$acc_now_delinq <- NULL

summary(loan_dt_sub$collections_12_mths_ex_med)
nrow(loan_dt_sub[collections_12_mths_ex_med==0])/nrow(loan_dt_sub)
loan_dt_sub$collections_12_mths_ex_med <- NULL


### Out of the remaininig 10 columns, 9 are more than 92% complete and
# they are relatively less skewed, so we can impute with median.
for(ii in c('delinq_2yrs', 'inq_last_6mths','open_acc','pub_rec','total_acc',
            'revol_util','tot_coll_amt','tot_cur_bal','total_rev_hi_lim','annual_inc')){
  median_value <- median(loan_dt_sub[[ii]], na.rm = TRUE)
  loan_dt_sub[is.na(eval(parse(text = ii))), c(ii) := median_value ]
}


# The variable "mths_since_last_delinq" has around 50% missing.
# We will use predictive imputation for this variable based on other numeric variables
num_col <- sapply(loan_dt_sub, is.numeric) 
num_col <- names(num_col[num_col])
loan_dt_num <- loan_dt_sub[,num_col, with=FALSE]

# Since the imputation takes a lot of processing time in pc, so I just kept m=1 and maxit=1
imputed <- mice(loan_dt_num, m=1, maxit = 1, method = 'pmm', seed = 500)
completeData <- as.data.table(complete(imputed,1))

#write.csv(completeData, 'C:/Kaggle/Lending_Club/missing_cmplt.csv', row.names = FALSE)
# Append the imputed field into the original data set
loan_dt_sub$mths_since_last_delinq <- completeData$mths_since_last_delinq
rm(loan_dt_num)

# The field 'emp_length' has some values as 'n/a', we will manully replace them with '<1 year'
loan_dt_sub[, emp_length:=
              ifelse(emp_length=='n/a','< 1 year',emp_length)]




### Feature engineering--------

#create a new variable as tenure from the earliest credit line as a proxy for credit experience.
loan_dt_sub[, cr_tenure := as.numeric(2017 - as.numeric(substr(earliest_cr_line,5,8)))]
loan_dt_sub$earliest_cr_line <- NULL 

# Create dummy variables for all the categorical variable.
# Variable sub_grade has too many level, there will be too many dummy variable for these
# which will increase width of the dataset significantly. SO, we will keep only the 'grade'.
loan_dt_sub$sub_grade <- NULL
for (ii in c('term', 'grade', 'emp_length','verification_status',
            'pymnt_plan',  'purpose', 'application_type','addr_state')){
  
  loan_dt_sub <- cbind(loan_dt_sub, dummy(loan_dt_sub[[ii]], sep = "_"))
  # remove original variables
  loan_dt_sub[[ii]] <- NULL
}

# Convert the dummy variables into factors. Also replace the spaces in the variable names with '_'
dummy_vars <- grep("loan_dt_sub", names(loan_dt_sub), value = TRUE)
for(ii in dummy_vars){
  loan_dt_sub[[ii]] <- as.factor(loan_dt_sub[[ii]])
  setnames(loan_dt_sub, ii, gsub(" ",'_',ii))
}

#cc<- copy(loan_dt_sub)
#loan_dt_sub <- cc

# remove special chatacter from the variable names
setnames(loan_dt_sub,'loan_dt_sub_<_1_year','loan_dt_sub_l_1_year')
setnames(loan_dt_sub,'loan_dt_sub_10+_years', 'loan_dt_sub_10p_years')


#### Feature Selection-------

## A) run a correlation analysis among numerical variables
num_col <- sapply(loan_dt_sub, is.numeric) 
num_col <- names(num_col[num_col])
loan_dt_num <- loan_dt_sub[,num_col, with=FALSE]

corr <- round(cor(loan_dt_num), 1)
ggcorrplot(corr)

# Take away from the correlation plot:
# 1)
# loan_amnt,funded_amnt,funded_amnt_inv, installment are highly correlated with
# each other. So for now, I am going to keep only the loan_amnt in the data set
# excluding other

# 2)
# total_rec_prncp,total_pymnt_inv,tota_pymnt are highly correlated.
# So, I am going to keep only the total_pymnt

loan_dt_sub[, c('funded_amnt','funded_amnt_inv', 'installment'):=NULL]
loan_dt_sub[,c('total_rec_prncp','total_pymnt_inv'):=NULL]


## B) feature importance from random forest classifier
loan_dt_sub <- na.omit(loan_dt_sub)
rf_classifier <- randomForest(target ~ ., data=loan_dt_sub, mtry=11, ntree=50, importance=TRUE)

# Feature importance
feat_imp <- importance(rf_classifier, type=2)

# Making a new column with rownames
names <- rownames(feat_imp)
feat_imp <- data.table(feat_imp)
feat_imp$names <- data.table(names)

# sort features in decreasing order of importance
feat_imp <- feat_imp[order(MeanDecreaseGini, decreasing = TRUE)]

#Due to memory constraint, I am just taking top 20 features to fit the model.
top20_feat <- feat_imp[1:20,][['names']]


### Model fitting----------

# Create modeling data set with just top 20 variables
mdl_dt <- loan_dt_sub[, c(top20_feat, 'target'), with=FALSE]

# Split training and test data
part_idx <- createDataPartition(mdl_dt$target, p=0.6,list = FALSE,times = 1)
mdl_train <- mdl_dt[part_idx,]
mdl_test <- mdl_dt[-part_idx,]

# setting the lavel of the target variable. As for xgboost lael must be in [0,num_class]
tr_label <- as.numeric(mdl_train$target)-1

# Train model
xgb_clf <- xgboost(data = as.matrix(mdl_train[, -c('target'), with=FALSE]), 
               label = tr_label, 
               eta = 0.1,
               max_depth = 15, 
               nround=5, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 2,
               nthread = 3
)

# Print training error for 5 iterations
print(xgb_clf$evaluation_log)

#predict with test data set
predictions <- predict(xgb_clf, as.matrix(mdl_test[, -c('target'), with=FALSE]))

# Convert the prediction probabilty vector into a data frame 2 columns,
# each for probabilty to class 0 and 1
class_prob <- cbind.data.frame(split(predictions, rep(1:2, times=length(predictions)/2)),
                               stringsAsFactors=F)
names(class_prob) <- c('prob_0','prob_1')

# If we set the threshold probability for class as 0.5
class_prob$score <- as.numeric(class_prob$prob_1>0.5)

# Appending the prediction to the test data set
mdl_test$prediction <- class_prob$score

# Test data Accuracy
print(sum(mdl_test$target==mdl_test$prediction)/nrow(mdl_test))
