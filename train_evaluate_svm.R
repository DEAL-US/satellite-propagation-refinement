# Load the necessary libraries
library(dplyr)
library(asteRisk)
library(asteRiskData)
getLatestSpaceData()
library(e1071)

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

data_aux <- select(data,
                   -ephemerisUTCTime_base,
                   -ephemerisUTCTime_predict,
                   -X_predict_real,
                   -Y_predict_real,
                   -Z_predict_real
)

a <- read.table(file = "error_models.txt", header = TRUE)

# Loop through each seed value provided in the 'seeds' variable
for (seed in seeds) {
  
  # Set the random seed to ensure reproducibility
  set.seed(seed)
  
  # Randomly sample half of the data indices for training
  sample <- sample(nrow(data), as.integer(0.5*nrow(data)))
  
  # Split non-sequential data into training and test sets and convert to tensors
  data_train  <- data_aux[sample,]
  data_test  <- data_aux[-sample,]
  
  data_X_train <- select(data_train,
                         -error_Y_predict,
                         -error_Z_predict)
  colnames(data_X_train)[colnames(data_X_train) == "error_X_predict"] <- "y"
  data_Y_train <- select(data_train,
                         -error_X_predict,
                         -error_Z_predict)
  colnames(data_Y_train)[colnames(data_Y_train) == "error_Y_predict"] <- "y"
  data_Z_train <- select(data_train,
                         -error_X_predict,
                         -error_Y_predict)
  colnames(data_Z_train)[colnames(data_Z_train) == "error_Z_predict"] <- "y"
  
  # Train the SVM model
  svm_X <- svm(
    y ~ ., data = data_X_train,
    type = "eps-regression",
    kernel = "radial",
    epsilon = 0.1,
    cost = 1,
    tolerance = 0.01,
  )
  svm_Y <- svm(
    y ~ ., data = data_Y_train,
    type = "eps-regression",
    kernel = "radial",
    epsilon = 0.1,
    cost = 1,
    tolerance = 0.01,
  )
  svm_Z <- svm(
    y ~ ., data = data_Z_train,
    type = "eps-regression",
    kernel = "radial",
    epsilon = 0.1,
    cost = 1,
    tolerance = 0.01,
  )
  
  
  
  data_X_test <- select(data_test,
                        -error_Y_predict,
                        -error_Z_predict)
  colnames(data_X_test)[colnames(data_X_test) == "error_X_predict"] <- "y"
  data_Y_test <- select(data_test,
                        -error_X_predict,
                        -error_Z_predict)
  colnames(data_Y_test)[colnames(data_Y_test) == "error_Y_predict"] <- "y"
  data_Z_test <- select(data_test,
                        -error_X_predict,
                        -error_Y_predict)
  colnames(data_Z_test)[colnames(data_Z_test) == "error_Z_predict"] <- "y"
  
  
  # Make predictions
  predictions_X <- predict(svm_X, newdata = data_X_test)
  predictions_Y <- predict(svm_Y, newdata = data_Y_test)
  predictions_Z <- predict(svm_Z, newdata = data_Z_test)
  
  
  # Evaluate the model
  non_corrected_predictions <- select(data_test,
                                      X_predict_sgdp4,
                                      Y_predict_sgdp4,
                                      Z_predict_sgdp4)
  non_corrected_predictions <- as.matrix(unname(non_corrected_predictions))
  
  error_predictions <- cbind(predictions_X, predictions_Y, predictions_Z)
  
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