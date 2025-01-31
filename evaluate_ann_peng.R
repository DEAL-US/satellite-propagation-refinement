# Load the necessary libraries
library(dplyr)
library(asteRisk)
library(asteRiskData)
getLatestSpaceData()
library(tensorflow)
library(keras)

source("utils.R")

# Define directories for data and models
data_directory <- "path_to_data_directory"
models_directory <- "path_to_models_directory"
model_name <- "ann_peng"

# Define the number of iterations and create a sequence of seeds for each iteration
n_iter <- 9
seeds <- c(667:(667 + n_iter))

# Set the working directory to the data directory and read the dataset
setwd(data_directory)
data <- read.table("dataset.txt", header = TRUE)

data$prediction_duration <- data$unixTime_predict-data$unixTime_base
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
               X_predict_sgdp4,
               Y_predict_sgdp4,
               Z_predict_sgdp4,
               dX_predict_sgdp4,
               dY_predict_sgdp4,
               dZ_predict_sgdp4,
               X_predict_real,
               Y_predict_real,
               Z_predict_real,
               error_X_predict,
               error_Y_predict,
               error_Z_predict
)

process_row <- function(row) {
  pos_base <- c(row["X_base_real"], row["Y_base_real"], row["Z_base_real"])
  vel_base <- c(row["dX_base_real"], row["dY_base_real"], row["dZ_base_real"])
  koe_base <- get_orbital_parameters(row["ephemerisUTCTime_base"], as.integer(pos_base), as.integer(vel_base))
  
  pos_predict <- c(row["X_predict_sgdp4"], row["Y_predict_sgdp4"], row["Z_predict_sgdp4"])
  vel_predict <- c(row["dX_predict_sgdp4"], row["dY_predict_sgdp4"], row["dZ_predict_sgdp4"])
  koe_predict <- get_orbital_parameters(row["ephemerisUTCTime_predict"], as.integer(pos_predict), as.integer(vel_predict))
  return(c(koe_base, koe_predict))
}

dim(data)
data_koe <- t(apply(data, 1, process_row))
data$n0_base <- data_koe[,1]
data$e0_base <- data_koe[,2]
data$i0_base <- data_koe[,3]
data$M0_base <- data_koe[,4]
data$omega0_base <- data_koe[,5]
data$OMEGA0_base <- data_koe[,6]

data$n0_predict <- data_koe[,7]
data$e0_predict <- data_koe[,8]
data$i0_predict <- data_koe[,9]
data$M0_predict <- data_koe[,10]
data$omega0_predict <- data_koe[,11]
data$OMEGA0_predict <- data_koe[,12]


a <- read.table(file = "error_models.txt", header = TRUE)

# Loop through each seed value provided in the 'seeds' variable
for (seed in seeds) {
  
  
  print(seed)
  
  # Set the random seed for TensorFlow operations to ensure reproducibility
  tensorflow::set_random_seed(seed)
  
  # Randomly sample half of the data indices for training
  sample <- sample(nrow(data), as.integer(0.5*nrow(data)))
  
  # Split non-sequential data into training and test sets and convert to tensors
  data_train  <- data[sample,]
  data_test  <- data[-sample,]
  
  x_train <- select(data_train,
                    -ephemerisUTCTime_base,
                    -ephemerisUTCTime_predict,
                    -X_predict_real,
                    -Y_predict_real,
                    -Z_predict_real,
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
                    -ephemerisUTCTime_base,
                    -ephemerisUTCTime_predict,
                    -X_predict_real,
                    -Y_predict_real,
                    -Z_predict_real,
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
  
  
  # Min-max normalization
  
  # Normalize training data
  min_x <- apply(x_train, 2, min)
  max_x <- apply(x_train, 2, max)
  x_train <- (x_train - min_x) / (max_x - min_x)
  
  # Normalize target variable for training data
  min_y_X <- apply(y_X_train, 2, min)
  max_y_X <- apply(y_X_train, 2, max)
  y_X_train <- (y_X_train - min_y_X) / (max_y_X - min_y_X)
  
  min_y_Y <- apply(y_Y_train, 2, min)
  max_y_Y <- apply(y_Y_train, 2, max)
  y_Y_train <- (y_Y_train - min_y_Y) / (max_y_Y - min_y_Y)
  
  min_y_Z <- apply(y_Z_train, 2, min)
  max_y_Z <- apply(y_Z_train, 2, max)
  y_Z_train <- (y_Z_train - min_y_Z) / (max_y_Z - min_y_Z)
  
  # Normalize test data using training data statistics
  x_test <- (x_test - min_x) / (max_x - min_x)
  y_X_test <- (y_X_test - min_y_X) / (max_y_X - min_y_X)
  y_Y_test <- (y_Y_test - min_y_Y) / (max_y_Y - min_y_Y)
  y_Z_test <- (y_Z_test - min_y_Z) / (max_y_Z - min_y_Z)
  
  
  # Load model
  
  d_test <- select(data[-sample,], 
                   ephemerisUTCTime_base,
                   ephemerisUTCTime_predict,
                   tiempo_pred_hours,
                   tiempo_pred_group,
                   X_base_real,
                   Y_base_real,
                   Z_base_real,
                   X_predict_real,
                   Y_predict_real,
                   Z_predict_real,
                   X_predict_sgdp4,
                   Y_predict_sgdp4,
                   Z_predict_sgdp4,
                   error_X_predict,
                   error_Y_predict,
                   error_Z_predict
  )
  
  setwd(models_directory)
  #Model X
  model_X <- load_model_hdf5(paste0(model_name,"_X_",seed,".hdf5"))
  
  error_predictions_normalized_X <- model_X %>% predict(x_test)
  # error_predictions_normalized[1,]
  
  
  error_predictions_X <- as_tensor(as.matrix(unname(error_predictions_normalized_X)))
  error_predictions_X <- error_predictions_X*(max_y_X - min_y_X) + min_y_X
  
  non_corrected_predictions_X <- select(data_test,
                                        X_predict_sgdp4)
  non_corrected_predictions_X <- as_tensor(as.matrix(unname(non_corrected_predictions_X)))
  
  corrected_predictions_X <- non_corrected_predictions_X + error_predictions_X
  corrected_predictions_X <- as.matrix(corrected_predictions_X)
  
  # Model Y
  model_Y <- load_model_hdf5(paste0(model_name,"_Y_",seed,".hdf5"))
  
  error_predictions_normalized_Y <- model_Y %>% predict(x_test)
  
  error_predictions_Y <- as_tensor(as.matrix(unname(error_predictions_normalized_Y)))
  error_predictions_Y <- error_predictions_Y*(max_y_Y - min_y_Y) + min_y_Y
  
  non_corrected_predictions_Y <- select(data_test,
                                        Y_predict_sgdp4)
  non_corrected_predictions_Y <- as_tensor(as.matrix(unname(non_corrected_predictions_Y)))
  
  corrected_predictions_Y <- non_corrected_predictions_Y + error_predictions_Y
  corrected_predictions_Y <- as.matrix(corrected_predictions_Y)
  
  
  #Model Z
  model_Z <- load_model_hdf5(paste0(model_name,"_Z_",seed,".hdf5"))
  
  error_predictions_normalized_Z <- model_Z %>% predict(x_test)
  
  error_predictions_Z <- as_tensor(as.matrix(unname(error_predictions_normalized_Z)))
  error_predictions_Z <- error_predictions_Z*(max_y_Z - min_y_Z) + min_y_Z
  
  non_corrected_predictions_Z <- select(data_test,
                                        Z_predict_sgdp4)
  non_corrected_predictions_Z <- as_tensor(as.matrix(unname(non_corrected_predictions_Z)))
  
  corrected_predictions_Z <- non_corrected_predictions_Z + error_predictions_Z
  corrected_predictions_Z <- as.matrix(corrected_predictions_Z)
  
  
  
  
  # Combine corrected predictions for X, Y, Z coordinates
  corrected_predictions <- data.frame("X" = corrected_predictions_X,
                                      "Y" = corrected_predictions_Y,
                                      "Z" = corrected_predictions_Z)
  
  
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