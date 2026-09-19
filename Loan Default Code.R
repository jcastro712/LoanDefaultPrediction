# Loan Default Prediction
# ECON484
# Machine Learning

setwd("~/Desktop/Projects/Loan Default prediction")

# Libraries
#####
library(tidyverse)
library(randomForest)
library(xgboost)
library(broom)
library(foreach)
library(doSNOW)
library(pROC)
library(scales)
#####

# Data
#####
rawdata <- read_csv("Loan_default.csv")
anyNA(rawdata)

# Holdout sample
set.seed(1234)
dataindex <- seq_len(nrow(rawdata))
holdout <- sample(dataindex,
                  size = floor(0.20*nrow(rawdata)),
                  replace = FALSE)
holdout_data <- rawdata[holdout,]
loan_data <- rawdata[-holdout,]

# Check proportions
datapos <- sum(rawdata$Default == 1)
dataneg <- sum(rawdata$Default == 0)
dataspw <- dataneg/datapos

holdpos <- sum(holdout_data$Default == 1)
holdneg <- sum(holdout_data$Default == 0)
holdspw <- holdneg/holdpos

loanpos <- sum(loan_data$Default == 1)
loanneg <- sum(loan_data$Default == 0)
loanspw <- loanneg/loanpos

# Train/Test Split
set.seed(1234)
train_size <- floor(0.80*nrow(loan_data))
train_idx <- sample(seq_len(nrow(loan_data)),
                    size = train_size,
                    replace = FALSE)

loan_train_raw <- loan_data[train_idx,]
loan_test_raw <- loan_data[-train_idx,]
#####

# Data Visualization
#####
data_plots <- rawdata %>%
  mutate(Default = factor(Default,
                          levels = c(0, 1),
                          labels = c("No Default", "Default")))

# Plots
ggplot(data_plots, aes(x = Default)) +
  geom_bar(fill = "#4FAFD6", alpha = 0.7) +
  geom_text(stat = "count",
            aes(label = percent(after_stat(count)/sum(after_stat(count)))),
            vjust = -0.3) +
  scale_y_continuous(labels=label_number(scale=1/1000,suffix='K')) +
  labs(title = "Default Rate (Class Balance)",
       x = "",
       y = "Count",
       caption = "Figure 1") +
  theme_classic() 

ggplot(data_plots, aes(x = Default, y = CreditScore, fill = Default)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("#CFE8FF", "#1E78A8")) +
  labs(title = "Credit Score by Default Status",
       x = "",
       y = "Credit Score",
       caption = "Figure 2") +
  theme_classic() +
  theme(legend.position = "none")

ggplot(data_plots, aes(x = Default, y = Income, fill = Default)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("#CFE8FF", "#1E78A8")) +
  scale_y_continuous(labels=label_number(scale=1/1000,suffix='K')) +
  labs(title = "Income by Default Status",
       x = "",
       y = "Income",
       caption = "Figure 3") +
  theme_classic() +
  theme(legend.position = "none")

ggplot(data_plots, aes(x = Default, y = InterestRate, fill = Default)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("#CFE8FF", "#1E78A8")) +
  labs(title = "Interest Rate by Default Status",
       x = "",
       y = "Interest Rate",
       caption = "Figure 4") +
  theme_classic() + 
  theme(legend.position = "none")
#####

# Functions to clean data
#####
clean_base <- function(loan_data) {
  loan_clean <- loan_data %>%
    select(-LoanID) %>%
    mutate(HasMortgage = ifelse(HasMortgage == "Yes", 1, 0),
           HasDependents = ifelse(HasDependents == "Yes", 1, 0),
           HasCoSigner = ifelse(HasCoSigner == "Yes", 1, 0),
           
           Education = as.factor(Education),
           EmploymentType = as.factor(EmploymentType),
           MaritalStatus = as.factor(MaritalStatus),
           LoanPurpose = as.factor(LoanPurpose))
  
  loan_clean
}

clean_logit_data <- function(loan_data) {
  logit_data <- clean_base(loan_data)
  return(logit_data)
}

clean_forest_data <- function(loan_data){
  forest_data <- clean_base(loan_data)
  forest_data$Default <- as.factor(forest_data$Default)
  return(forest_data)
}

clean_xgboost_data <- function(loan_data){
  xgb_data <- clean_base(loan_data)
  X <- model.matrix(Default ~ . - 1,
                    data = xgb_data)
  xgb_data <- data.frame(Default = as.numeric(xgb_data$Default),
                         X,
                         check.names = FALSE)
  return(xgb_data)
}
#####

# Logit Model
#####
# Clean data
logit_train <- clean_logit_data(loan_train_raw)
logit_test <- clean_logit_data(loan_test_raw)

# Train logit model
logit_model <- glm(Default ~ ., 
                   data = logit_train,
                   family = binomial(link="logit"))

# Metrics
logit_prob <- predict(logit_model,
                      newdata=logit_test,
                      type="response")
logit_mse <- mean((logit_prob - logit_test$Default)^2)
logit_mse

# Importance Matrix
logit_top10 <- tidy(logit_model) %>%
  filter(term != "(Intercept)") %>%
  mutate(Importance = abs(statistic)) %>%
  slice_max(Importance, n=10) %>%
  mutate(Model = "Logit",
         Variable = reorder(term, Importance)) %>%
  select(Model, Variable, Importance)

logit_plot <- logit_top10 %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#A8D8F0") +
  labs(title = "Logit Model – Most Important Variables",
       x = "Importance (Z-stat)",
       y = "Variable",
       caption = "Figure 5") +
  theme_classic()

logit_plot
#####

# Random Forest
#####
# Clean data
forest_train <- clean_forest_data(loan_train_raw)
forest_test <- clean_forest_data(loan_test_raw)

# Train Random Forest model
forest_model <- randomForest(formula=Default ~ .,
                             data=forest_train,
                             ntree=200,
                             mtry=5,
                             importance = TRUE)

# Metrics
forest_prob <- predict(forest_model, 
                       forest_test, 
                       type="prob")[,"1"]
forest_y <- as.numeric(as.character(forest_test$Default))

forest_mse <- mean((forest_prob - forest_y)^2)
forest_mse

# Importance Matrix
forest_importance <- importance(forest_model)

forest_top10 <- data.frame(Variable = rownames(forest_importance),
                           Importance = forest_importance[, "MeanDecreaseGini"],
                           row.names = NULL) %>%
  arrange(desc(Importance)) %>%
  slice_head(n=10) %>%
  mutate(Model = "Random Forest",
         Variable = reorder(Variable, Importance))

forest_plot <- forest_top10 %>%
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#4FA3D9", alpha = 0.9) +
  labs(title = "Random Forest – Most Important Variables",
       x = "Importance (Mean Decrease Gini)",
       y = "Variable",
       caption = "Figure 6") +
  theme_classic()

forest_plot
#####

# XGBoost
#####
# Clean Data
xgboost_train <- clean_xgboost_data(loan_train_raw)
xgboost_test <- clean_xgboost_data(loan_test_raw)

# Extract target variable
train_label <- xgboost_train$Default
test_label <- xgboost_test$Default

# Remove target variables from data
xgboost_train_matrix <- as.matrix(subset(xgboost_train, select= -Default))
xgboost_test_matrix <- as.matrix(subset(xgboost_test, select= -Default))

# Train XGBoost Model
xgb_model <- xgboost(data = xgboost_train_matrix,
                    label = train_label,
                    max_depth = 3,
                    eta = 0.1,
                    nrounds = 100,  
                    lambda = 1,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    verbose = 0)

# Metrics
xgb_prob <- predict(xgb_model, xgboost_test_matrix)
xgb_mse <- mean((xgb_prob - test_label)^2)

# Importance Matrix
xgb_importance <- xgb.importance(feature_names = colnames(xgboost_train_matrix),
                                 model = xgb_model)

xgb_top10 <- xgb_importance %>%
  slice_max(Gain, n=10) %>%
  mutate(Model = "XGBoost",
         Feature = reorder(Feature, Gain))

xgb_plot <- xgb_top10 %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_col(fill = "#2F86C4") +
  labs(title = "XGBoost – Most Important Variables",
       x = "Importance (Gain)",
       y = "Variable",
       caption = "Figure 7") +
  theme_classic()

xgb_plot
#####

# Model Tuning
#####
# Define parameter grid
max_depth_values <- c(2, 3, 4, 5)
lambda_values <- c(0.5, 1, 1.5, 2)
eta_values <- c(0.05, 0.075, 0.1, 0.15)

grid <- expand.grid(max_depth = max_depth_values,
                    lambda = lambda_values,
                    eta = eta_values)

# Parallel Setup
num_cores <- max(1, parallel::detectCores() - 2)
cl <- parallel::makeCluster(num_cores)
registerDoSNOW(cl)

# Progress Bar
pb <- txtProgressBar(min = 0,
                     max = nrow(grid),
                     style = 3)
progress <- function(n) setTxtProgressBar(pb,n)
opts <- list(progress = progress)

# Start Loop
loopstart <- Sys.time()

  xgb_results <- foreach(i = 1:nrow(grid),
                      .combine = rbind,
                      .packages = c("xgboost"),
                      .options.snow = opts,
                      .export = c("xgboost_train_matrix", "xgboost_test_matrix", 
                                  "train_label", "test_label")
                      ) %dopar% {
                        
    params <- grid[i,]
    
    dtrain <- xgb.DMatrix(data = xgboost_train_matrix, label = train_label)
    dtest <- xgb.DMatrix(data = xgboost_test_matrix, label = test_label)
    
    # Train XGBoost model with current parameters
    xgb_params <- list(booster = "gbtree",
                       objective = "binary:logistic",
                       max_depth = params$max_depth,
                       eta = params$eta,
                       lambda = params$lambda,
                       nthread = 1)
    
    xgb_model <- xgb.train(params = xgb_params,
                           data = dtrain,
                           nrounds = 1000,
                           watchlist = list(val = dtest),
                           early_stopping_rounds = 30,
                           verbose = 0)
    
    probs <- predict(xgb_model, xgboost_test_matrix)
    mse <- mean((probs - test_label)^2)

    data.frame(max_depth = params$max_depth,
               nrounds = xgb_model$best_iteration,
               lambda = params$lambda,
               eta = params$eta,
               mse = mse)
}

loopend <- Sys.time()

close(pb)
parallel::stopCluster(cl)

cat("Total Minutes:", as.numeric(difftime(loopend,loopstart, units = "min")),"\n")

best_params <- xgb_results[which.min(xgb_results$mse),]
print(best_params)
#####

# Final Model
#####
final_model <- xgboost(data = xgboost_train_matrix,
                       label = train_label,
                       max_depth = 2,
                       nrounds = 385,  
                       lambda = 1,
                       eta = 0.15,
                       booster = "gbtree",
                       objective = "binary:logistic",
                       verbose = 0)

best_xgb_predict <- predict(final_model, xgboost_test_matrix)

# MSE
best_xgb_mse <- mean((best_xgb_predict - test_label)^2)
best_xgb_mse

# Importance Matrix
final_importance <- xgb.importance(feature_names = colnames(xgboost_train_matrix),
                                   model = final_model)

final_top10 <- final_importance %>%
  slice_max(Gain, n=10) %>%
  mutate(Model = "XGBoost",
         Feature = reorder(Feature, Gain))

final_plot <- final_top10 %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_col(fill = "#1F6FAE") +
  labs(title = "Tuned XGBoost – Most Important Variables",
       x = "Importance (Gain)",
       y = "Variable",
       caption = "Figure 8") +
  theme_classic()

final_plot
#####

# Results
#####
logit_pred <- ifelse(logit_prob >= 0.5, 1, 0)
forest_pred <- ifelse(forest_prob >= 0.5, 1, 0)
xgb_pred <- ifelse(xgb_prob >= 0.5, 1, 0)
best_xgb_pred <- ifelse(best_xgb_predict >= 0.5, 1, 0)

# Create function
calculate_metrics <- function(actual, predicted, prob) {
  TP <- sum(predicted==1 & actual==1)
  TN <- sum(predicted==0 & actual==0)
  FP <- sum(predicted==1 & actual==0)
  FN <- sum(predicted==0 & actual==1)
  
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  precision <- TP/(TP+FP)
  recall <- TP/(TP+FN)
  auc <- as.numeric(auc(actual,predicted))
  
  data.frame(Accuracy = accuracy,
             Precision = precision,
             Recall = recall,
             AUC = auc)
}

logit_metrics <- calculate_metrics(actual=logit_test$Default,
                                   predicted=logit_pred,
                                   prob=logit_prob)

forest_metrics <- calculate_metrics(actual=forest_y,
                                    predicted=forest_pred,
                                    prob=xgb_prob)

xgb_metrics <- calculate_metrics(actual=test_label,
                                 predicted=xgb_pred,
                                 prob=xgb_prob)

best_xgb_metrics <- calculate_metrics(actual=test_label,
                                      predicted=best_xgb_pred,
                                      prob=best_xgb_predict)

results <- data.frame(Model = c("Logit", "RandomForest", "XGBoost", "Tuned XGBoost"),
                      MSE = c(logit_mse, forest_mse, xgb_mse, best_xgb_mse))
print(results)
#####

# Final Performance
#####
# Clean Data
holdout_logit <-clean_logit_data(holdout_data)
holdout_forest <- clean_forest_data(holdout_data)
holdout_xgb <- clean_xgboost_data(holdout_data)

y_hold_logit <- holdout_logit$Default
y_hold_forest <- as.numeric(as.character(holdout_forest$Default))
y_hold_xgb <- holdout_clean$Default

# Metrics
# Logit
logit_hold_prob <- predict(logit_model,
                           newdata=holdout_logit,
                           type="response")
logit_hold_mse <- mean((logit_hold_prob - y_hold_logit)^2)

logit_hold_pred <- ifelse(logit_hold_prob >= 0.5, 1, 0)
logit_hold_metrics <- calculate_metrics(actual=y_hold_logit,
                                        predicted=logit_hold_pred,
                                        prob=logit_hold_prob)

# Random Forest
forest_hold_prob <- predict(forest_model,
                            newdata=holdout_forest,
                            type="prob")[,"1"]
forest_hold_mse <- mean((forest_hold_prob - y_hold_forest)^2)

forest_hold_pred <- ifelse(forest_hold_prob >= 0.5, 1, 0)
forest_hold_metrics <- calculate_metrics(actual=y_hold_forest,
                                         predicted=forest_hold_pred,
                                         prob=forest_hold_prob)

# Untuned XGBoost
x_holdout <- as.matrix(subset(holdout_clean, select = -Default))

old_hold_xgb_prob <- predict(xgb_model, x_holdout)
old_hold_xgb_mse <- mean((old_hold_xgb_prob - y_hold_xgb)^2)

xgb_old_pred <- ifelse(old_hold_xgb_prob >= 0.5, 1, 0)
xgb_old_metrics <- calculate_metrics(actual=y_hold_xgb,
                                     predicted=xgb_old_pred,
                                     prob=old_hold_xgb_prob)

# Tuned XGBoost
holdout_prob <- predict(final_model, x_holdout)
holdout_mse <- mean((holdout_prob - y_hold_xgb)^2)

xgb_hold_pred <- ifelse(holdout_prob >= 0.5, 1, 0)

xgb_hold_metrics <- calculate_metrics(actual=y_hold_xgb,
                                      predicted=xgb_hold_pred,
                                      prob=holdout_prob)

#####
