library(dplyr)
library(keras)
library(tensorflow)

# Define directories for data and models
data_directory <- "path_to_data_directory"
models_directory <- "path_to_models_directory"
model_name <- "smj + pos + vel"

# Define the number of iterations and create a sequence of seeds for each iteration
n_iter <- 9
seeds <- c(667:(667+n_iter))

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

# Select features for the model input
x <- select(data,
            unixTime_base,
            X_base_real,             
            Y_base_real,
            Z_base_real,
            dX_base_real,
            dY_base_real,
            dZ_base_real,
            X_Sun,
            Y_Sun,
            Z_Sun,
            # X_Mercury,
            # Y_Mercury,
            # Z_Mercury,
            # X_Venus,
            # Y_Venus,
            # Z_Venus,
            X_Moon,
            Y_Moon,
            Z_Moon,
            # lunar_libration_Phi,
            # lunar_libration_Theta,
            # lunar_libration_Psi,
            # X_Mars,
            # Y_Mars,
            # Z_Mars,
            X_Jupiter,
            Y_Jupiter,
            Z_Jupiter,
            # X_Saturn,
            # Y_Saturn,
            # Z_Saturn,
            # X_Uranus,
            # Y_Uranus,
            # Z_Uranus,
            # X_Neptune,
            # Y_Neptune,
            # Z_Neptune,
            # X_Pluto,
            # Y_Pluto,
            # Z_Pluto,
            unixTime_predict,
            # mov_X_sgdp4,
            # mov_Y_sgdp4,
            # mov_Z_sgdp4
            X_predict_sgdp4,
            Y_predict_sgdp4,
            Z_predict_sgdp4
)
x <- as.matrix(unname(x))

# Select target variables for the model output
y <- select(data,
            error_X_predict,
            error_Y_predict,
            error_Z_predict)
y <- as.matrix(unname(y))

# Read a file containing error models, to be updated with new results
a <- read.table(file = "error_models.txt", header = TRUE)
c <- a

# Loop through each seed for model training and evaluation
for (seed in seeds) {
  
  # Set the random seed for TensorFlow
  tensorflow::set_random_seed(seed)
  
  # Randomly sample half of the data for training
  sample <- sample(nrow(data), as.integer(0.5 * nrow(data)))
  
  # Split data into training and testing sets
  x_train <- x[sample,]
  x_train <- as_tensor(x_train)
  y_train <- y[sample,]
  y_train <- as_tensor(y_train)
  
  x_test  <- x[-sample,]
  x_test <- as_tensor(x_test)
  y_test <- y[-sample,]
  y_test <- as_tensor(y_test)
  
  # Normalize training data using IQR and median
  iqr_x <- apply(x_train, 2, IQR)
  median_x <- apply(x_train, 2, median)
  x_train <- (x_train-median_x) / iqr_x
  
  iqr_y <- apply(y_train, 2, IQR)
  median_y <- apply(y_train, 2, median)
  y_train <- (y_train-median_y) / iqr_y
  
  # Normalize testing data using the same parameters as training data
  x_test <- (x_test-median_x) / iqr_x
  y_test <- (y_test-median_y) / iqr_y
  
  # Load the model corresponding to the current seed
  setwd(models_directory)
  model <- load_model_hdf5(paste0(model_name, "_", seed, ".hdf5"))
  
  # Evaluate the model on the test set
  result <- model %>% evaluate(x_test, y_test)

  # Predict errors on the test set
  error_predictions_normalized <- model %>% predict(x_test)

  # Convert normalized predictions back to original scale
  error_predictions <- as_tensor(as.matrix(unname(error_predictions_normalized)),
                                 shape = shape(nrow(error_predictions_normalized), 3))
  error_predictions <- error_predictions*iqr_y + median_y

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

  # Calculate corrected predictions by adding predicted errors to SGDP4 predictions
  non_corrected_predictions <- select(d_test,                            
                                      X_predict_sgdp4,
                                      Y_predict_sgdp4,
                                      Z_predict_sgdp4)

  non_corrected_predictions <- as.matrix(unname(non_corrected_predictions))
  non_corrected_predictions <- as_tensor(non_corrected_predictions,
                                         shape = shape(nrow(non_corrected_predictions),3))

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