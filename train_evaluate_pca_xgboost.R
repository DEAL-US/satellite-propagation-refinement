# Load the necessary libraries
library(dplyr)
library(asteRisk)
library(asteRiskData)
getLatestSpaceData()
library(caret)
library(xgboost)
library(mlr)
library(tensorflow)

source("utils.R")

# Define the number of iterations and create a sequence of seeds for each iteration
n_iter <- 9
seeds <- c(667:(667 + n_iter))


# Define directories for data and models
data_directory <- "path_to_data_directory"

# Set the working directory to the data directory and read the dataset
setwd(data_directory)
data <- read.table("dataset.tx", header = TRUE)

data$prediction_duration <- data$unixTime_predict-data$unixTime_base
data$mov_X_sgdp4 <- data$X_predict_sgdp4 - data$X_base_real
data$mov_Y_sgdp4 <- data$Y_predict_sgdp4 - data$Y_base_real
data$mov_Z_sgdp4 <- data$Z_predict_sgdp4 - data$Z_base_real
data$error_X_predict <- data$X_predict_real-data$X_predict_sgdp4
data$error_Y_predict <- data$Y_predict_real-data$Y_predict_sgdp4
data$error_Z_predict <- data$Z_predict_real-data$Z_predict_sgdp4

# Select data for the model
data <- select(data,
               ephemerisUTCTime_base,
               ephemerisUTCTime_predict,
               tiempo_pred_hours,
               tiempo_pred_group,
               prediction_duration,
               X_base_real,             
               Y_base_real,
               Z_base_real,
               dX_base_real,
               dY_base_real,
               dZ_base_real,
               X_predict_real,
               Y_predict_real,
               Z_predict_real,
               X_predict_sgdp4,
               Y_predict_sgdp4,
               Z_predict_sgdp4,
               dX_predict_sgdp4,
               dY_predict_sgdp4,
               dZ_predict_sgdp4,
               error_X_predict,
               error_Y_predict,
               error_Z_predict
)

# Function to get the input parameters
process_row <- function(row) {
  pos_base <- as.integer(c(row["X_base_real"], row["Y_base_real"], row["Z_base_real"]))
  vel_base <- as.integer(c(row["dX_base_real"], row["dY_base_real"], row["dZ_base_real"]))
  
  pos_predict <- as.integer(c(row["X_predict_sgdp4"], row["Y_predict_sgdp4"], row["Z_predict_sgdp4"]))
  vel_predict <- as.integer(c(row["dX_predict_sgdp4"], row["dY_predict_sgdp4"], row["dZ_predict_sgdp4"]))
  
  koe_base <- get_orbital_parameters(row["ephemerisUTCTime_base"], pos_base, vel_base)
  koe_predict <- get_orbital_parameters(row["ephemerisUTCTime_predict"], pos_predict, vel_predict)
  
  mov_koe <- koe_predict - koe_base
  
  mov_rtn <- transform_to_rtn(pos_base, vel_base, pos_predict)
  
  return(c(koe_base, mov_koe, mov_rtn))
}

data_expand <- t(apply(data, 1, process_row))
data$n0_base <- data_expand[,1]
data$e0_base <- data_expand[,2]
data$i0_base <- data_expand[,3]
data$M0_base <- data_expand[,4]
data$omega0_base <- data_expand[,5]
data$OMEGA0_base <- data_expand[,6]

data$mov_n0 <- data_expand[,7]
data$mov_e0 <- data_expand[,8]
data$mov_i0 <- data_expand[,9]
data$mov_M0 <- data_expand[,10]
data$mov_omega0 <- data_expand[,11]
data$mov_OMEGA0 <- data_expand[,12]

data$mov_R <- data_expand[,13]
data$mov_T <- data_expand[,14]
data$mov_N <- data_expand[,15]

data_aux <- select(data,
                   prediction_duration,
                   n0_base,
                   e0_base,
                   i0_base,
                   M0_base,
                   omega0_base,
                   OMEGA0_base,
                   mov_n0,
                   mov_e0,
                   mov_i0,
                   mov_M0,
                   mov_omega0,
                   mov_OMEGA0,
                   mov_R,
                   mov_T,
                   mov_N,
                   error_X_predict,
                   error_Y_predict,
                   error_Z_predict
)

a <- read.table(file = "error_models.txt", header = TRUE)

for (seed in seeds) {
  
  print(seed)
  
  # Set the random seed to ensure reproducibility
  tensorflow::set_random_seed(seed)
  
  # Randomly sample half of the data indices for training
  sample <- sample(nrow(data), as.integer(0.5*nrow(data)))
  
  # Split non-sequential data into training and test sets and convert to tensors
  data_train  <- data_aux[sample,]
  data_test  <- data_aux[-sample,]
  
  x_train <- select(data_train,
                    -error_X_predict,
                    -error_Y_predict,
                    -error_Z_predict)
  x_train <- as_tensor(as.matrix(unname(x_train)))
  y_X_train <- select(data_train,
                      error_X_predict)
  y_X_train <- as_tensor(as.matrix(unname(y_X_train)))
  y_Y_train <- select(data_train,
                      error_Y_predict)
  y_Y_train <- as_tensor(as.matrix(unname(y_Y_train)))
  y_Z_train <- select(data_train,
                      error_Z_predict)
  y_Z_train <- as_tensor(as.matrix(unname(y_Z_train)))
  
  x_test  <- select(data_test,
                    -error_X_predict,
                    -error_Y_predict,
                    -error_Z_predict)
  x_test <- as_tensor(as.matrix(unname(x_test)))
  y_X_test <- select(data_test,
                     error_X_predict)
  y_X_test <- as_tensor(as.matrix(unname(y_X_test)))
  y_Y_test <- select(data_test,
                     error_Y_predict)
  y_Y_test <- as_tensor(as.matrix(unname(y_Y_test)))
  y_Z_test <- select(data_test,
                     error_Z_predict)
  y_Z_test <- as_tensor(as.matrix(unname(y_Z_test)))
  
  ###################
  ## Normalizaci?n ##
  ###################
  
  # Min-max normalization
  
  # Normalize training data
  min_x <- apply(x_train, 2, min)
  max_x <- apply(x_train, 2, max)
  x_train_norm <- (x_train - min_x) / (max_x - min_x)
  
  # Normalize target variable for training data
  min_y_X <- apply(y_X_train, 2, min)
  max_y_X <- apply(y_X_train, 2, max)
  y_X_train_norm <- (y_X_train - min_y_X) / (max_y_X - min_y_X)
  
  min_y_Y <- apply(y_Y_train, 2, min)
  max_y_Y <- apply(y_Y_train, 2, max)
  y_Y_train_norm <- (y_Y_train - min_y_Y) / (max_y_Y - min_y_Y)
  
  min_y_Z <- apply(y_Z_train, 2, min)
  max_y_Z <- apply(y_Z_train, 2, max)
  y_Z_train_norm <- (y_Z_train - min_y_Z) / (max_y_Z - min_y_Z)
  
  # Normalize test data using training data statistics
  x_test_norm <- (x_test - min_x) / (max_x - min_x)
  y_X_test_norm <- (y_X_test - min_y_X) / (max_y_X - min_y_X)
  y_Y_test_norm <- (y_Y_test - min_y_Y) / (max_y_Y - min_y_Y)
  y_Z_test_norm <- (y_Z_test - min_y_Z) / (max_y_Z - min_y_Z)
  
  # Apply PCA
  pca_model <- prcomp(x_train_norm, center = TRUE, scale. = TRUE)
  explained_variance <- cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2)
  
  # Select number of components to retain 95% variance
  num_components <- which.min(abs(explained_variance - 0.95))
  
  # Transform into selected components
  x_train_pca <- predict(pca_model, x_train_norm)[, 1:num_components]
  x_test_pca <- predict(pca_model, x_test_norm)[, 1:num_components]
  
  
  # Prepare data for XGBoost
  train_matrix_X <- xgb.DMatrix(data = as.matrix(x_train_pca), label = as.matrix(y_X_train_norm))
  test_matrix_X <- xgb.DMatrix(data = as.matrix(x_test_pca), label = as.matrix(y_X_test_norm))
  
  train_matrix_Y <- xgb.DMatrix(data = as.matrix(x_train_pca), label = as.matrix(y_Y_train_norm))
  test_matrix_Y <- xgb.DMatrix(data = as.matrix(x_test_pca), label = as.matrix(y_Y_test_norm))
  
  train_matrix_Z <- xgb.DMatrix(data = as.matrix(x_train_pca), label = as.matrix(y_Z_train_norm))
  test_matrix_Z <- xgb.DMatrix(data = as.matrix(x_test_pca), label = as.matrix(y_Z_test_norm))
  
  # Set XGBoost parameters
  params <- list(
    booster = "gbtree",
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = 0.05,
    max_depth = 4,
    min_child_weight = 7,
    subsample = 0.5,
    colsample_bytree = 0.6,
    lambda = 0.8,
    alpha = 0.1
  )
  
  # Train the model
  print("Model X")
  xgb_model_X <- xgb.train(
    params = params,
    data = train_matrix_X,
    nrounds = 5000,
    watchlist = list(train = train_matrix_X),
    early_stopping_rounds = 50,
    verbose = 0
  )
  print("Model Y")
  xgb_model_Y <- xgb.train(
    params = params,
    data = train_matrix_Y,
    nrounds = 5000,
    watchlist = list(train = train_matrix_Y),
    early_stopping_rounds = 50,
    verbose = 0
  )
  print("Model Z")
  xgb_model_Z <- xgb.train(
    params = params,
    data = train_matrix_Z,
    nrounds = 5000,
    watchlist = list(train = train_matrix_Z),
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  error_predictions_normalized_X <- as.matrix(predict(xgb_model_X, test_matrix_X))
  error_predictions_normalized_X <- as_tensor(as.matrix(unname(error_predictions_normalized_X)))
  error_predictions_X <- error_predictions_normalized_X*(max_y_X - min_y_X) + min_y_X
  
  error_predictions_normalized_Y <- as.matrix(predict(xgb_model_Y, test_matrix_Y))
  error_predictions_normalized_Y <- as_tensor(as.matrix(unname(error_predictions_normalized_Y)))
  error_predictions_Y <- error_predictions_normalized_Y*(max_y_Y - min_y_Y) + min_y_Y
  
  error_predictions_normalized_Z <- as.matrix(predict(xgb_model_Z, test_matrix_Z))
  error_predictions_normalized_Z <- as_tensor(as.matrix(unname(error_predictions_normalized_Z)))
  error_predictions_Z <- error_predictions_normalized_Z*(max_y_Z - min_y_Z) + min_y_Z
  
  error_predictions <- cbind(error_predictions_X, error_predictions_Y, error_predictions_Z)
  
  d_test <- data[-sample,]
  
  non_corrected_predictions <- select(d_test,
                                      X_predict_sgdp4,
                                      Y_predict_sgdp4,
                                      Z_predict_sgdp4)
  non_corrected_predictions <- as.matrix(unname(non_corrected_predictions))
  
  
  corrected_predictions <- non_corrected_predictions + error_predictions
  corrected_predictions <- as.matrix(corrected_predictions)
  corrected_predictions <- as.data.frame(corrected_predictions)
  colnames(corrected_predictions) <- c("X","Y","Z")
  
  # Create a data frame with model predictions and errors
  b <- data.frame("id" = as.numeric(rownames(d_test)),
                  "model" = model_name,
                  "seed" = seed,
                  "base_date" = d_test$ephemerisUTCTime_base,
                  "prediction_date" = d_test$ephemerisUTCTime_predict,
                  "hours_pred" = d_test$hours_pred,
                  "time_pred_group" = d_test$time_pred_group,
                  "base_X" = d_test$X_base_real,
                  "base_Y" = d_test$Y_base_real,
                  "base_Z" = d_test$Z_base_real,
                  "real_pred_X" = d_test$X_predict_real,
                  "real_pred_Y" = d_test$Y_predict_real,
                  "real_pred_Z" = d_test$Z_predict_real,
                  "prediction_X" = corrected_predictions$X,
                  "prediction_Y" = corrected_predictions$Y,
                  "prediction_Z" = corrected_predictions$Z,
                  "error_X" = d_test$X_predict_real-corrected_predictions$X,
                  "error_Y" = d_test$Y_predict_real-corrected_predictions$Y,
                  "error_Z" = d_test$Z_predict_real-corrected_predictions$Z
  )
  b$spatial_error <- sqrt(b$error_X^2 + b$error_Y^2 + b$error_Z^2)
  
  a <- rbind(a,b)
}

setwd(data_directory)
write.table(a, file = "error_models.txt", row.names = FALSE, col.names = TRUE)