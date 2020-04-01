library(dplyr)

setwd("C:/Users/Yashar/Desktop/Data Science Bootcamp/R programming/Week 8")
#=================================
#       Data Pre-processing
#=================================

# Load some packages for data manipulation: 
library(tidyverse)
library(magrittr)

# Clear workspace: 
rm(list = ls())



library(caret)
boston_dataset <- read.csv('Boston.csv')
boston_dataset$chas <- h2o.asfactor(boston_dataset$chas)
set.seed(1)
id <- createDataPartition(y = boston_dataset$chas, p = 0.7, list = FALSE)
df_train <- boston_dataset[id, ]
df_test <- boston_dataset[-id, ]


# Activate h2o package for using: 
library(h2o)
h2o.init(nthreads = 16, max_mem_size = "16g")

h2o.no_progress()

# Convert to h2o Frame and identify inputs and output: 
test <- as.h2o(df_test)
train <- as.h2o(df_train)

response <- "chas"
predictors <- setdiff(names(train), response)


#===================================
#   Pure, Lasso and Ridge Logistic
#===================================

# Train Logistic Model: 
pure_logistic <- h2o.glm(family= "binomial", 
                         x = predictors, 
                         y = response, 
                         lambda = 0, 
                         training_frame = train)


# Function Shows the coefficients table: 

show_coeffs <- function(model_selected) {
  model_selected@model$coefficients_table %>% 
    as.data.frame() %>% 
    mutate_if(is.numeric, function(x) {round(x, 3)}) %>% 
    filter(coefficients != 0) %>% 
    knitr::kable()
}


# Use this function: 
show_coeffs(pure_logistic)

# Lasso Logistic Model: 

lasso_logistic <- h2o.glm(family = "binomial", 
                          alpha = 1,
                          seed = 1988, 
                          x = predictors, 
                          y = response, 
                          training_frame = train)

show_coeffs(lasso_logistic)

# Ridge Logistic Model: 
ridge_logistic <- h2o.glm(family = "binomial", 
                          alpha = 0,
                          seed = 1988, 
                          x = predictors, 
                          y = response, 
                          training_frame = train)

show_coeffs(ridge_logistic)

# Function shows model performance on test data: 

my_cm <- function(model_selected) {
  pred <- h2o.predict(model_selected, test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, df_test$chas) %>% 
    return()
}

lapply(list(pure_logistic, lasso_logistic, ridge_logistic), my_cm)

#All models give the same outputs