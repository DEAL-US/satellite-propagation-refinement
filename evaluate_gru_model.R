library(dplyr)
library(keras)
library(tensorflow)

# Define directories for data and models
data_directory <- "path_to_data_directory"
models_directory <- "path_to_models_directory"
model_name <- "gru(svmmj) + mov + vel"

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

# Select non-sequential data for the current prediction time
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

# Select error data for the current prediction time
y <- select(data,
            error_X_predict,
            error_Y_predict,
            error_Z_predict)
y <- as.matrix(unname(y))

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
  y_train <- y[sample,]
  
  non_seq_data_test  <- non_seq_data[-sample,]
  non_seq_data_test <- as_tensor(non_seq_data_test)
  seq_data_test <- seq_data[-sample,]
  seq_data_test <- as_tensor(seq_data_test, shape = c(nrow(seq_data_test), 10, n_col))
  y_test <- y[-sample,]
  
  # Normalize training data
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train) * 10, n_col))
  iqr_seq <- apply(seq_data_train, 2, IQR)
  median_seq <- apply(seq_data_train, 2, median)
  seq_data_train <- (seq_data_train - median_seq) / iqr_seq
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train) / 10, 10, n_col))
  
  iqr_non_seq <- apply(non_seq_data_train, 2, IQR)
  median_non_seq <- apply(non_seq_data_train, 2, median)
  non_seq_data_train <- (non_seq_data_train - median_non_seq) / iqr_non_seq
  
  iqr_y <- IQR(y_train)
  median_y <- median(y_train)
  y_train <- (y_train - median_y) / iqr_y
  y_train <- as_tensor(y_train)
  
  # Normalize test data
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test) * 10, n_col))
  seq_data_test <- (seq_data_test - median_seq) / iqr_seq
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test) / 10, 10, n_col))
  
  non_seq_data_test <- (non_seq_data_test - median_non_seq) / iqr_non_seq
  
  y_test <- (y_test - median_y) / iqr_y
  y_test <- as_tensor(y_test)
  
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
  
  # Load and evaluate model
  setwd(models_directory)
  model <- load_model_hdf5(paste0(model,"_",seed,".hdf5"))
  
  result <- model %>% evaluate(list(seq = seq_data_test, non_seq = non_seq_data_test),
                                 y_test
  )
  
  error_predictions_normalized <- model %>% predict(list(seq = seq_data_test, non_seq = non_seq_data_test))
  
  error_predictions <- as_tensor(as.matrix(unname(error_predictions_normalized)),
                                   shape = shape(nrow(error_predictions_normalized), 1)
  )
  error_predictions_X <- error_predictions * iqr_y + median_y
  
  non_corrected_predictions <- select(d_test, X_predict_sgdp4)
  non_corrected_predictions <- as.matrix(unname(non_corrected_predictions))
  non_corrected_predictions <- as_tensor(non_corrected_predictions,
                                         shape = shape(nrow(non_corrected_predictions), 1))
  
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
  
  # Append the results to the cumulative data frame
  c <- rbind(c, b)
  
}

# Save the updated error models to a file
setwd(data_directory)
write.table(c, file = "error_models.txt", row.names = FALSE, col.names = TRUE)
