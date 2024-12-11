###################################################
###################################################
#######       Student Capstone Project      #######
#######             Chirui GUO              #######
#######              PH125.9x               #######
#######             2024/12/11              #######
###################################################
###################################################

#############################
###    Packages Loading   ###
#############################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggtext)) install.packages("ggtext", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(smotefamily)) install.packages("smotefamily", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(ggtext)
library(caret)
library(gbm)
library(smotefamily)
library(doParallel)


#############################
###    Dataset Loading    ###
#############################

options(timeout = 120)
dl <- "student.zip"
if(!file.exists(dl))
  download.file("https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip", dl)
data <- "data.csv"
if(!file.exists(data))
  unzip(dl, data)

df <- read_delim(data, delim=";", show_col_types = FALSE)
rm(data, dl)


################################
###   Exploratory Analysis   ###
################################

# Explore Data

df |>
  group_by(`Target`) |>
  ggplot(aes(x = `Target`)) +
  geom_bar() +
  geom_text(stat = 'count', 
            aes(label = after_stat(count)),
            vjust = -0.5) +
  labs(title = "Target Distribution") +
  theme_minimal()



################################
###           Models         ###
################################

# transform data type
df <- df |>
  mutate(across(where(is.factor), as.numeric))|>  # transform factor into numeric
  mutate(across(where(is.character), as.factor)) # transform character into factor

# Set random seed
set.seed(42)

# 1.Data preprocessing
# Split train set and test set
index <- createDataPartition(df$Target, p = 0.8, list = FALSE)
train_data <- df[index, ]
test_data <- df[-index, ]

# 2. Applying ADASYN algorithm and SMOTE algorithm
# prepared train set
X_train <- train_data |> 
  select(-Target)
y_train <- train_data |> 
  select(Target)

# use SMOTE, ADASYN and Borderline-SMOTE to balance data
SMOTE_balanced <- SMOTE(X_train, y_train, K = 5)
ADASYN_balanced <- ADAS(X_train, y_train, K = 5)
BLSMOTE_balanced <- BLSMOTE(X_train, y_train, K = 5)

# get tibble
SMOTE_balanced_train <- SMOTE_balanced$data |>
  rename(Target = class)
ADASYN_balanced_train <- ADASYN_balanced$data |>
  rename(Target = class)
BLSMOTE_balanced_train <- BLSMOTE_balanced$data |>
  rename(Target = class)

rm(df, X_train, y_train, SMOTE_balanced, ADASYN_balanced, BLSMOTE_balanced) # free memory
# 3. Train the model
# use the balanced data to train the model

## Cross validation is too slow so we should do parallel computing
## Detect the number of cores and create a parallel cluster.
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# choose gradient boosting as machine learning model
ML_model <- "gbm"

Baseline_model <- train(
  Target ~ .,
  data = train_data,
  method = ML_model,
  trControl = trainControl(
    method = "cv",
    number = 5
  )
)

SMOTE_model <- train(
  Target ~ .,
  data = SMOTE_balanced_train,
  method = ML_model,
  trControl = trainControl(
    method = "cv",
    number = 5
  )
)

ADASYN_model <- train(
  Target ~ .,
  data = ADASYN_balanced_train,
  method = ML_model,
  trControl = trainControl(
    method = "cv",
    number = 5
  )
)

BLSMOTE_model <- train(
  Target ~ .,
  data = BLSMOTE_balanced_train,
  method = ML_model,
  trControl = trainControl(
    method = "cv",
    number = 5
  )
)

stopCluster(cl) #stop
registerDoSEQ() # recover sequential computing
rm(cl) # free the memory

# 4. Prediction and Evaluation

## Baseline
predictions <- predict(Baseline_model, test_data)
cm <- confusionMatrix(predictions, as.factor(test_data$Target))
accuracy <- as.numeric(cm$overall["Accuracy"]) # get accuracy
f1_scores <- cm$byClass[, "F1"]
macro_f1 <- mean(f1_scores, na.rm = TRUE) # get f1 score
f1_results <- tibble(method = "Baseline", accuracy = accuracy, f1 = macro_f1)

## SMOTE
predictions <- predict(SMOTE_model, test_data)
cm <- confusionMatrix(predictions, as.factor(test_data$Target))
accuracy <- as.numeric(cm$overall["Accuracy"]) # get accuracy
f1_scores <- cm$byClass[, "F1"]
macro_f1 <- mean(f1_scores, na.rm = TRUE) # get f1 score
f1_results <- bind_rows(f1_results, tibble(method="SMOTE", accuracy = accuracy, f1 = macro_f1))

## ADASYN
predictions <- predict(ADASYN_model, test_data)
cm <- confusionMatrix(predictions, as.factor(test_data$Target))
accuracy <- as.numeric(cm$overall["Accuracy"]) # get accuracy
f1_scores <- cm$byClass[, "F1"]
macro_f1 <- mean(f1_scores, na.rm = TRUE) # get f1 score
f1_results <- bind_rows(f1_results, tibble(method="ADASYN", accuracy = accuracy, f1 = macro_f1))

## BLSMOTE
predictions <- predict(BLSMOTE_model, test_data)
cm <- confusionMatrix(predictions, as.factor(test_data$Target))
accuracy <- as.numeric(cm$overall["Accuracy"]) # get accuracy
f1_scores <- cm$byClass[, "F1"]
macro_f1 <- mean(f1_scores, na.rm = TRUE) # get f1 score
f1_results <- bind_rows(f1_results, tibble(method="BLSMOTE", accuracy = accuracy, f1 = macro_f1))

f1_results
