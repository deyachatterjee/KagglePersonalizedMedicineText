rm(list=ls())
library(needs)
needs(readr,dplyr,tidyr,ggplot2,xgboost,caret,proxy,logisticPCA,rARPACK,ica,tibble,stringr,forcats,corrplot,lubridate,mlr,data.table)
needs(MASS,class,e1071,tm,SnowballC,Matrix,syuzhet,nnet,gsubfn,reshape2,devtools)
needs(qdapTools)
training_variants=fread("/media/suresh/520beb03-066a-471e-bdc1-e91782049d99/kaggle/PersonalizedMedicine/data/training_variants")
testing_variants=fread("/media/suresh/520beb03-066a-471e-bdc1-e91782049d99/kaggle/PersonalizedMedicine/data/test_variants")
train_txt_dump <- tibble(text = read_lines("/media/suresh/520beb03-066a-471e-bdc1-e91782049d99/kaggle/PersonalizedMedicine/data/training_text", skip = 1))
train_txt <- train_txt_dump %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
train_txt <- train_txt %>%
  mutate(ID = as.integer(ID))

test_txt_dump <- tibble(text = read_lines("/media/suresh/520beb03-066a-471e-bdc1-e91782049d99/kaggle/PersonalizedMedicine/data/test_text", skip = 1))
test_txt <- test_txt_dump %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
test_txt <- test_txt %>%
  mutate(ID = as.integer(ID))

labelCountEncoding <- function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}


# Merge training and test sets with text data
trainfull <- merge(training_variants, train_txt, by="ID")
testfull <- merge(testing_variants, test_txt, by="ID")

Classes <- trainfull$Class
trainfull$Class <- NULL

# Combine all data
fulldata <- rbind(trainfull, testfull)

# Basic text features (character and word #)
fulldata$nchar <- as.numeric(nchar(fulldata$txt))
fulldata$nwords <- as.numeric(str_count(fulldata$txt, "\\S+"))

# TF-IDF (uncomment when running)
txt1 <- Corpus(VectorSource(fulldata$txt))
txt1 <- tm_map(txt1, stripWhitespace)
txt1 <- tm_map(txt1, content_transformer(tolower))
txt1 <- tm_map(txt1, removePunctuation)
txt1 <- tm_map(txt1, removeWords, stopwords("english"))
txt1 <- tm_map(txt1, stemDocument, language="english")
txt1 <- tm_map(txt1, removeNumbers)
dtm <- DocumentTermMatrix(txt1, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.95)

# Create dataframe
fulldata <- cbind(fulldata, as.matrix(dtm))
fulldata_backup <- fulldata

#### Used as restarting point when trying out new features

fulldata <- fulldata_backup

# Split first letter off variation for typical variations
fulldata <- fulldata %>% 
  extract(Variation, into=c("First_Letter", "Var_2"), 
          regex = "^(?=.{1,7}$)([a-zA-Z]+)([0-9].*)$", 
          remove = FALSE)

# Split number and last letter for typical variations
fulldata <- fulldata %>% 
  extract(Var_2, into=c("Gene_Location", "Last_Letter"),
          regex = "^([0-9]+)([a-zA-Z]|.*)$",
          remove = TRUE)

# Identify and encode deletions
fulldata$is_del <- ifelse(grepl("del", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode insertions
fulldata$is_ins <- ifelse(grepl("ins", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode fusions
fulldata$is_fus <- ifelse(grepl("fus", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode truncation
fulldata$is_trunc <- ifelse(grepl("trunc", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode methylations
fulldata$is_methyl <- ifelse(grepl("methyl", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode amplifications
fulldata$is_amp <- ifelse(grepl("amp", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode silencing
fulldata$is_sil <- ifelse(grepl("sil", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode overexpression
fulldata$is_expr <- ifelse(grepl("expr", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode splicing
fulldata$is_splice <- ifelse(grepl("splice", fulldata$Variation, ignore.case = T), 1, 0)

# Identify and encode exon variations
fulldata$is_exon <- ifelse(grepl("exon", fulldata$Variation, ignore.case = T), 1, 0)
# One hot encode first letter variables
firstletter_encoded <- mtabulate(fulldata$First_Letter)
colnames(firstletter_encoded) <- paste("first_", colnames(firstletter_encoded), sep="")
firstletter_encoded <- subset(firstletter_encoded, select=-c(first_CASP, first_DNMT))

# One hot encode last letter variables
lastletter_encoded <- mtabulate(fulldata$Last_Letter)
colnames(lastletter_encoded) <- paste("last_", colnames(lastletter_encoded), sep="")
lastletter_encoded <- subset(lastletter_encoded, select=-c(last_BRAF, last_del, last_dup))

# Identify and encode normal variations (Letter-Number-Letter) (using last letter)
lastletter_encoded$normal_vars <- rowSums(lastletter_encoded[,])

# Bind first and last letter vars back in
fulldata <- cbind(firstletter_encoded, fulldata)
fulldata <- cbind(lastletter_encoded, fulldata)

# Identify and encode named variations (Deletion, insertion, etc.)
fulldata$named_vars <- rowSums(fulldata[,c("is_del", "is_ins", "is_fus", "is_trunc", "is_methyl", "is_amp", "is_sil", "is_expr", "is_splice", "is_exon")])
# Identify and encode wierd variations (anything not above)
fulldata$weird_vars <- rowSums(fulldata[,c("normal_vars", "named_vars")])
fulldata$weird_vars <- ifelse(fulldata$weird_vars == 0, 1, 0)

# Change Gene and variation to factors for label encoding
fulldata$Gene <- as.factor(fulldata$Gene)
fulldata$Variation <- as.factor(fulldata$Variation)

# Label count encoding
fulldata$Gene <- labelCountEncoding(fulldata$Gene)
fulldata$Variation <- labelCountEncoding(fulldata$Variation)
fulldata <- subset(fulldata, select=-c(First_Letter, Last_Letter))
fulldata[is.na(fulldata)] <- 0
traindata <- fulldata[1:3321,]
testdata <- fulldata[3322:8989,]
train_ID <- traindata$ID
Classes <- Classes - 1
traindata$ID <- NULL
traindata$txt <- NULL
test_ID <- testdata$ID
testdata$ID <- NULL
testdata$txt <- NULL
# Create matrices
traindata[] <- lapply(traindata, as.numeric)
testdata[] <- lapply(testdata, as.numeric)
dtrain <- xgb.DMatrix(Matrix(as.matrix(traindata), sparse = TRUE), label = Classes)
dtest <- xgb.DMatrix(Matrix(as.matrix(testdata), sparse = TRUE))

# XGBoost parameters
set.seed(6457)
num_class <- 9
xgb_params_txt <- list(colsample_bytree = 0.7, 
                       subsample = 0.7, 
                       booster = "gbtree",
                       max_depth = 4, 
                       eta = 0.05, 
                       eval_metric = "mlogloss", 
                       objective = "multi:softprob",
                       gamma = 1,
                       num_class = num_class)

# Find optimal # of rounds (uncomment when running)
xgb_cv_txt <- xgb.cv(xgb_params_txt, dtrain, early_stopping_rounds = 10, nfold = 5, nrounds=3000)
gb_dt_txt <- xgb.train(xgb_params_txt, dtrain, nrounds = 180)
importance_matrix <- xgb.importance(model = gb_dt_txt)
xgb.plot.importance(importance_matrix = importance_matrix)
test_txt_predict <- predict(gb_dt_txt, dtest)
test_txt_predict <- t(matrix(test_txt_predict, nrow=9, ncol=length(test_txt_predict)/9))
test_txt_predict <- as.data.frame(test_txt_predict)
names(test_txt_predict)[1:9] <- c("class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9")
test_txt_predict <- cbind(test_ID, test_txt_predict)
names(test_txt_predict)[1] <- "ID"
test_txt_predict$ID <- as.integer(test_txt_predict$ID)
test_txt_predict %>% write_csv('/media/suresh/520beb03-066a-471e-bdc1-e91782049d99/kaggle/PersonalizedMedicine/Predictions/Cancer_base_predictions_withsentiment.csv')