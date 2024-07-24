library(dplyr)
library(keras)
library(tensorflow)

# Define directories for data and models
data_directory <- "path_to_data_directory"
models_directory <- "path_to_models_directory"
model_name <- "gru(svmmj) + mov + vel (XYZ)"

# Define the number of iterations and create a sequence of seeds for each iteration
n_iter <- 9
seeds <- c(667:(667 + n_iter))

# Set the working directory to the data directory and read the dataset
setwd(data_directory)
data <- read.table("dataset.txt", header = TRUE)

# Calculate movement and prediction errors in X, Y, Z coordinates
data$mov_X_sgdp4 <- data$X_predict_sgdp4 - data$X_base_real
data$mov_Y_sgdp4 <- data$Y_predict_sgdp4 - data$Y_base_real
data$mov_Z_sgdp4 <- data$Z_predict_sgdp4 - data$Z_base_real
data$error_X_predict <- data$X_predict_real-data$X_predict_sgdp4
data$error_Y_predict <- data$Y_predict_real-data$Y_predict_sgdp4
data$error_Z_predict <- data$Z_predict_real-data$Z_predict_sgdp4

# Read error models data
a <- read.table(file = "error_models.txt", header = TRUE)

# Select non-sequential data
non_seq_data <- select(data,
                       unixTime_base,
                       X_base_real,             
                       Y_base_real,
                       Z_base_real,
                       dX_base_real,
                       dY_base_real,
                       dZ_base_real,
                       unixTime_predict,
                       mov_X_sgdp4,
                       mov_Y_sgdp4,
                       mov_Z_sgdp4
                      #  X_predict_sgdp4,
                      #  Y_predict_sgdp4,
                      #  Z_predict_sgdp4
)

non_seq_data <- as.matrix(unname(non_seq_data))

# Select error data
y_X <- select(data,
              error_X_predict)
y_X <- as.matrix(unname(y_X))
y_Y <- select(data,
              error_Y_predict)
y_Y <- as.matrix(unname(y_Y))
y_Z <- select(data,
              error_Z_predict)
y_Z <- as.matrix(unname(y_Z))

# Read sequential data and set the number of columns
seq_data <- as.matrix(read.table("sequential_data_svmmj.txt"))
n_col <- 15 # 9 for smj and 15 for svmmj

# Loop over each seed
for (seed in seeds) {
  
  # Initialize dataframe for current seed
  c <- data.frame()
  
  # Set random seed and create training and test samples
  tensorflow::set_random_seed(seed)
  sample <- sample(nrow(non_seq_data), as.integer(0.5 * nrow(non_seq_data)))
  
  non_seq_data_train  <- non_seq_data[sample,]
  non_seq_data_train <- as_tensor(non_seq_data_train)
  seq_data_train <- seq_data[sample,]
  seq_data_train <- as_tensor(seq_data_train, shape = c(nrow(seq_data_train), 10, n_col))
  y_X_train <- y_X[sample,]
  y_Y_train <- y_Y[sample,]
  y_Z_train <- y_Z[sample,]
  
  non_seq_data_test  <- non_seq_data[-sample,]
  non_seq_data_test <- as_tensor(non_seq_data_test)
  seq_data_test <- seq_data[-sample,]
  seq_data_test <- as_tensor(seq_data_test, shape = c(nrow(seq_data_test), 10, n_col))
  y_X_test <- y_X[-sample,]
  y_Y_test <- y_Y[-sample,]
  y_Z_test <- y_Z[-sample,]
  
  # Normalize training data
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train) * 10, n_col))
  iqr_seq <- apply(seq_data_train, 2, IQR)
  median_seq <- apply(seq_data_train, 2, median)
  seq_data_train <- (seq_data_train - median_seq) / iqr_seq
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train) / 10, 10, n_col))
  
  iqr_non_seq <- apply(non_seq_data_train, 2, IQR)
  median_non_seq <- apply(non_seq_data_train, 2, median)
  non_seq_data_train <- (non_seq_data_train - median_non_seq) / iqr_non_seq
  
  iqr_y_X <- IQR(y_X_train)
  median_y_X <- median(y_X_train)
  y_X_train <- (y_X_train - median_y_X) / iqr_y_X
  y_X_train <- as_tensor(y_X_train)
  
  iqr_y_Y <- IQR(y_Y_train)
  median_y_Y <- median(y_Y_train)
  y_Y_train <- (y_Y_train - median_y_Y) / iqr_y_Y
  y_Y_train <- as_tensor(y_Y_train)
  
  iqr_y_Z <- IQR(y_Z_train)
  median_y_Z <- median(y_Z_train)
  y_Z_train <- (y_Z_train - median_y_Z) / iqr_y_Z
  y_Z_train <- as_tensor(y_Z_train)
  
  # Normalize test data
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test) * 10, n_col))
  seq_data_test <- (seq_data_test - median_seq) / iqr_seq
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test) / 10, 10, n_col))
  
  non_seq_data_test <- (non_seq_data_test - median_non_seq) / iqr_non_seq
  
  y_X_test <- (y_X_test - median_y_X) / iqr_y_X
  y_X_test <- as_tensor(y_X_test)
  
  y_Y_test <- (y_Y_test - median_y_Y) / iqr_y_Y
  y_Y_test <- as_tensor(y_Y_test)
  
  y_Z_test <- (y_Z_test - median_y_Z) / iqr_y_Z
  y_Z_test <- as_tensor(y_Z_test)
  
  d_test <- select(data[-sample,], 
                   ephemerisUTCTime_base,
                   ephemerisUTCTime_predict,
                   hours_pred,
                   time_pred_group,
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
  
  # Load and evaluate model for X coordinate
  setwd(models_directory)
  model_X <- load_model_hdf5(paste0(model,"_X_",seed,".hdf5"))
  
  result <- model_X %>% evaluate(list(seq = seq_data_test, non_seq = non_seq_data_test),
                                 y_X_test
  )
  
  error_predictions_normalized_X <- model_X %>% predict(list(seq = seq_data_test, non_seq = non_seq_data_test))
  
  error_predictions_X <- as_tensor(as.matrix(unname(error_predictions_normalized_X)),
                                   shape = shape(nrow(error_predictions_normalized_X), 1)
  )
  error_predictions_X <- error_predictions_X * iqr_y_X + median_y_X
  
  non_corrected_predictions <- select(d_test, X_predict_sgdp4)
  non_corrected_predictions <- as.matrix(unname(non_corrected_predictions))
  non_corrected_predictions <- as_tensor(non_corrected_predictions,
                                         shape = shape(nrow(non_corrected_predictions), 1))
  
  corrected_predictions_X <- non_corrected_predictions + error_predictions_X
  corrected_predictions_X <- as.matrix(corrected_predictions_X)
  
  # Load and evaluate model for Y coordinate
  model_Y <- load_model_hdf5(paste0(model,"_Y_",seed,".hdf5"))
  
  result <- model_Y %>% evaluate(list(seq = seq_data_test, non_seq = non_seq_data_test),
                                 y_Y_test
  )
  
  error_predictions_normalized_Y <- model_Y %>% predict(list(seq = seq_data_test, non_seq = non_seq_data_test))
  
  error_predictions_Y <- as_tensor(as.matrix(unname(error_predictions_normalized_Y)),
                                   shape = shape(nrow(error_predictions_normalized_Y), 1)
  )
  error_predictions_Y <- error_predictions_Y * iqr_y_Y + median_y_Y
  
  non_corrected_predictions_Y <- select(d_test, Y_predict_sgdp4)
  non_corrected_predictions_Y <- as.matrix(unname(non_corrected_predictions_Y))
  non_corrected_predictions_Y <- as_tensor(non_corrected_predictions_Y,
                                           shape = shape(nrow(non_corrected_predictions_Y), 1))
  
  corrected_predictions_Y <- non_corrected_predictions_Y + error_predictions_Y
  corrected_predictions_Y <- as.matrix(corrected_predictions_Y)
  
  # Load and evaluate model for Z coordinate
  model_Z <- load_model_hdf5(paste0(model,"_Z_",seed,".hdf5"))
  
  result <- model_Z %>% evaluate(list(seq = seq_data_test, non_seq = non_seq_data_test),
                                 y_Z_test
  )
  
  error_predictions_normalized_Z <- model_Z %>% predict(list(seq = seq_data_test, non_seq = non_seq_data_test))
  
  error_predictions_Z <- as_tensor(as.matrix(unname(error_predictions_normalized_Z)),
                                   shape = shape(nrow(error_predictions_normalized_Z), 1)
  )
  error_predictions_Z <- error_predictions_Z * iqr_y_Z + median_y_Z
  
  non_corrected_predictions_Z <- select(d_test, Z_predict_sgdp4)
  non_corrected_predictions_Z <- as.matrix(unname(non_corrected_predictions_Z))
  non_corrected_predictions_Z <- as_tensor(non_corrected_predictions_Z,
                                           shape = shape(nrow(non_corrected_predictions_Z), 1))
  
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
                  "prediction_X" = predicciones_corregidas$X,
                  "prediction_Y" = predicciones_corregidas$Y,
                  "prediction_Z" = predicciones_corregidas$Z,
                  "error_X" = d_test$X_predict_real-predicciones_corregidas$X,
                  "error_Y" = d_test$Y_predict_real-predicciones_corregidas$Y,
                  "error_Z" = d_test$Z_predict_real-predicciones_corregidas$Z
    )
  b$spatial_error <- sqrt(b$error_X^2 + b$error_Y^2 + b$error_Z^2)
  
  # Append the results to the cumulative data frame
  c <- rbind(c, b)
  
}

# Save the updated error models to a file
setwd(data_directory)
write.table(c, file = "error_models.txt", row.names = FALSE, col.names = TRUE)
