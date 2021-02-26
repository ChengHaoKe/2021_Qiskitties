rm(list = ls())
header <- new.env()
source("/Users/Brian/git/2021_Qiskitties/header.R", local = header)

# Import 

encoded_data <- read_csv(header$int_data("encoded_data.csv")) %>% select(-X1)
resp_data    <- read_csv(header$int_data("y_data.csv")) %>% select(-X1)

# Combine

data <- encoded_data %>%
  bind_cols(resp_data) %>%
  clean_names() %>% 
  select(-manner_of_death_shot_and_tasered)

# Baseline model: threat level

library(caret)
library(nnet)

mfit    <- multinom(as.factor(threat_level) ~ ., data = data)

preds   <- as.numeric(as.factor(predict(mfit)))

actuals <- as.numeric(as.factor(data$threat_level)) 

sum(preds == actuals)/nrow(data) ## Calculate the fraction of correct predictions (ie: accuracy)

a <- summary(mfit)
b <- varImp(mfit)

# Baseline model: shooting frequency

shooting <- read_csv(header$int_data("ShootingFrequency.csv"))

library(caret)
library(nnet)

mfit2    <- multinom(Level ~ ., data = shooting)

preds2   <- predict(mfit2)

actuals2 <- shooting$Level

sum(preds2 == actuals2)/nrow(shooting) ## Calculate the fraction of correct predictions (ie: accuracy)

a2 <- summary(mfit2)
b2 <- varImp(mfit2)
