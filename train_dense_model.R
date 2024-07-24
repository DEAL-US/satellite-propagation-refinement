library(dplyr)
library(keras)
library(tensorflow)

# Define directories for data and models
data_directory <- "path_to_data_directory"
models_directory <- "path_to_models_directory"
model_name <- "planets + lib + pos + vel"

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

# Select non-sequential features for the model input
non_seq_data <- select(data,
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
            X_Mercury,
            Y_Mercury,
            Z_Mercury,
            X_Venus,
            Y_Venus,
            Z_Venus,
            X_Moon,
            Y_Moon,
            Z_Moon,
            lunar_libration_Phi,
            lunar_libration_Theta,
            lunar_libration_Psi,
            X_Mars,
            Y_Mars,
            Z_Mars,
            X_Jupiter,
            Y_Jupiter,
            Z_Jupiter,
            X_Saturn,
            Y_Saturn,
            Z_Saturn,
            X_Uranus,
            Y_Uranus,
            Z_Uranus,
            X_Neptune,
            Y_Neptune,
            Z_Neptune,
            X_Pluto,
            Y_Pluto,
            Z_Pluto,
            unixTime_predict,
            X_predict_sgdp4,
            Y_predict_sgdp4,
            Z_predict_sgdp4
)
non_seq_data <- as.matrix(unname(non_seq_data))

# Select the target variables
y <- select(data,
            error_X_predict,
            error_Y_predict,
            error_Z_predict)
y <- as.matrix(unname(y))

# Loop on seeds
for (seed in seeds) {
  
  # Set the random seed for reproducibility
  tensorflow::set_random_seed(seed)
  
  # Randomly sample half of the data indices for training
  sample <- sample(nrow(data), as.integer(0.5*nrow(data)))
  
  # Split data into training and test sets for both features and target variables
  non_seq_data_train <- non_seq_data[sample,]
  non_seq_data_train <- as_tensor(non_seq_data_train)
  y_train <- y[sample,]
  y_train <- as_tensor(y_train)
  
  non_seq_data_test  <- non_seq_data[-sample,]
  non_seq_data_test <- as_tensor(non_seq_data_test)
  y_test <- y[-sample,]
  y_test <- as_tensor(y_test)
  
  # Normalize training data using IQR and median
  iqr_non_seq_data <- apply(non_seq_data_train, 2, IQR)
  non_seq_datmediana_non_seq_data <- apply(non_seq_data_train, 2, median)
  non_seq_data_train <- (non_seq_data_train-non_seq_datmediana_non_seq_data) / iqr_non_seq_data
  
  iqr_y <- apply(y_train, 2, IQR)
  non_seq_datmediana_y <- apply(y_train, 2, median)
  y_train <- (y_train-non_seq_datmediana_y) / iqr_y
  
  # Normalize test data using training data statistics
  non_seq_data_test <- (non_seq_data_test-non_seq_datmediana_non_seq_data) / iqr_non_seq_data
  y_test <- (y_test-non_seq_datmediana_y) / iqr_y
  
  # Define the model architecture
  num_neurons_input <- ncol(non_seq_data_train)
  input <- layer_input(shape = num_neurons_input, name = "seq")
  
  x <- layer_dense(units = 128, activation = "relu")(input)
  x <- x + layer_dense(units = 128, activation = "relu")(x)
  x <- layer_dense(units = 256, activation = "relu")(x)
  x <- x + layer_dense(units = 256, activation = "relu")(x)
  x <- layer_dense(units = 128, activation = "relu")(x)
  x <- x + layer_dense(units = 128, activation = "relu")(x)
  x <- layer_dense(units = 64, activation = "relu")(x)
  
  # Output layer for prediction errors in X, Y, Z coordinates
  err_pred <- layer_dense(name = "error_pred", units = 3)(x)
  
  # Instantiate and compile the model
  model <- keras_model(inputs = input, outputs = err_pred)
  model %>% compile(optimizer = "adam", loss = "mse", metrics = "mae")
  
  setwd(models_directory)
  
  # Train the model
  history <- model %>% fit(
    x = non_seq_data_train,
    y = y_train,
    batch_size = 128,
    epochs = 500,
    callbacks = list(callback_model_checkpoint(paste0(model_name,"_",seed,".hdf5"),
                                               monitor = "mae",
                                               mode = "min",
                                               save_best_only = TRUE)
    )
  )
}