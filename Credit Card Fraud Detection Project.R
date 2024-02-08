#Credit Card Fraud project for HarvardX chose my own project.
# Yousef Waiel Said 

# Install and load required packages
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(kableExtra)) install.packages("kableExtra")
if (!require(tidyr)) install.packages("tidyr")
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(stringr)) install.packages("stringr")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(gbm)) install.packages("gbm")
if (!require(dplyr)) install.packages("dplyr")
if (!require(caret)) install.packages("caret")
if (!require(xgboost)) install.packages("xgboost")
if (!require(e1071)) install.packages("e1071")
if (!require(class)) install.packages("class")
if (!require(ROCR)) install.packages("ROCR")
if (!require(randomForest)) install.packages("randomForest")
if (!require(PRROC)) install.packages("PRROC")
if (!require(reshape2)) install.packages("reshape2")
if (!require(corrplot)) install.packages("corrplot")

# Load required libraries
library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(gbm)
library(caret)
library(xgboost)
library(e1071)
library(class)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)
library(corrplot)


# The data available in kaggle as the most rated datasete.
# The dataset (as a .csv files) can be downloaded at the following link:
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# download card's dataset to your pc.
# Loading the dataset as a .csv file on local system.
# I will save it as mydataset for the credit card data set.
# Load the credit card fraud dataset from Kaggle
mydataset <- read.csv("creditcard.csv")

# Set seed for reproducibility
set.seed(13)

# Create training and test datasets using stratified sampling
train_index <- createDataPartition(
  y = mydataset$Class,
  p = .6,
  list = F)

train <- mydataset[train_index,]
test_cv <- mydataset[-train_index,]

test_index <- createDataPartition(
  y = test_cv$Class,
  p = .5,
  list = F)

test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

# Remove unnecessary objects to free up memory
rm(train_index, test_index, test_cv)

#### K-Nearest Neighbors (KNN) Model ####

# Train KNN model with k=5
set.seed(13)
knn_model <- knn(train[, -30],
                 test[, -30],
                 train$Class,
                 k = 5,
                 prob = TRUE)

# Evaluate the model using AUC and AUPRC
pred <- prediction(
  as.numeric(as.character(knn_model)),
  as.numeric(as.character(test$Class)))

auc_val_knn <- performance(pred, "auc")
auc_plot_knn <- performance(pred, 'sens', 'spec')
auprc_plot_knn <- performance(pred, "prec", "rec")

# Compute AUPRC curve
auprc_val_knn <- pr.curve(
  scores.class0 = knn_model[test$Class == 1],
  scores.class1 = knn_model[test$Class == 0],
  curve = T,
  dg.compute = T)

# Plot AUC and AUPRC curves
plot(auc_plot_knn, main = paste("AUC:", auc_val_knn@y.values[[1]]))
plot(auprc_plot_knn, main = paste("AUPRC:", auprc_val_knn$auc.integral))
plot(auprc_val_knn)

#### Random Forest Model ####

# Train Random Forest model with 500 trees
set.seed(13)
rf_model <- randomForest(Class ~ ., data = train, ntree = 500)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = test)

# Evaluate the model using AUC and AUPRC
pred <- prediction(
  as.numeric(as.character(predictions)),
  as.numeric(as.character(test$Class)))

auc_val_rf <- performance(pred, "auc")
auc_plot_rf <- performance(pred, 'sens', 'spec')
auprc_plot_rf <- performance(pred, "prec", "rec",
                             curve = T,
                             dg.compute = T)
auprc_val_rf <- pr.curve(scores.class0 = predictions[test$Class == 1],
                         scores.class1 = predictions[test$Class == 0],
                         curve = T,
                         dg.compute = T)

# Plot AUC and AUPRC curves
plot(auc_plot_rf, main = paste("AUC:", auc_val_rf@y.values[[1]]))
plot(auprc_plot_rf, main = paste("AUPRC:", auprc_val_rf$auc.integral))
plot(auprc_val_rf)

# Display results in a table
results <- data.frame(
  Model = c("K-Nearest Neighbors", "Random Forest"),
  AUC = c(auc_val_knn@y.values[[1]], auc_val_rf@y.values[[1]]),
  AUPRC = c(auprc_val_knn$auc.integral, auprc_val_rf$auc.integral)
)

results

# Print results table using kable
results %>%
  kable() %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed", "responsive"),
    position = "center",
    font_size = 14,
    full_width = FALSE)
