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
                        # mov_X_sgdp4,
                        # mov_Y_sgdp4,
                        # mov_Z_sgdp4
                        X_predict_sgdp4,
                        Y_predict_sgdp4,
                        Z_predict_sgdp4
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
  
  # Set random seed and create training and test samples
  tensorflow::set_random_seed(seed)
  sample <- sample(nrow(data), as.integer(0.5*nrow(data)))
  
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
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train)*10, n_col))
  iqr_seq <- apply(seq_data_train, 2, IQR)
  median_seq <- apply(seq_data_train, 2, median)
  seq_data_train <- (seq_data_train-median_seq) / iqr_seq
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train)/10, 10, n_col))
  
  iqr_non_seq <- apply(non_seq_data_train, 2, IQR)
  median_non_seq <- apply(non_seq_data_train, 2, median)
  non_seq_data_train <- (non_seq_data_train-median_non_seq) / iqr_non_seq
  
  # Normalize target variables for training set
  iqr_y_X <- IQR(y_X_train)
  median_y_X <- median(y_X_train)
  y_X_train <- (y_X_train-median_y_X) / iqr_y_X
  y_X_train <- as_tensor(y_X_train)
  
  iqr_y_Y <- IQR(y_Y_train)
  median_y_Y <- median(y_Y_train)
  y_Y_train <- (y_Y_train-median_y_Y) / iqr_y_Y
  y_Y_train <- as_tensor(y_Y_train)
  
  iqr_y_Z <- IQR(y_Z_train)
  median_y_Z <- median(y_Z_train)
  y_Z_train <- (y_Z_train-median_y_Z) / iqr_y_Z
  y_Z_train <- as_tensor(y_Z_train)
  
  
  # Normalize the testing data using training data statistics
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test)*10, n_col))
  seq_data_test <- (seq_data_test-median_seq) / iqr_seq
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test)/10, 10, n_col))
  
  non_seq_data_test <- (non_seq_data_test-median_non_seq) / iqr_non_seq
  
  y_X_test <- (y_X_test-median_y_X) / iqr_y_X
  y_X_test <- as_tensor(y_X_test)
  
  y_Y_test <- (y_Y_test-median_y_Y) / iqr_y_Y
  y_Y_test <- as_tensor(y_Y_test)
  
  y_Z_test <- (y_Z_test-median_y_Z) / iqr_y_Z
  y_Z_test <- as_tensor(y_Z_test)
  
  
  # Define the model architecture for predicting errors in X, Y, Z coordinates
  seq_input <- layer_input(shape = c(10, n_col), name = "seq")  # Input layer for sequential data
  non_seq_input <- layer_input(shape = c(ncol(non_seq_data)), name = "non_seq")  # Input layer for non-sequential data
  
  # Process sequential and non-sequential inputs separately
  seq <- seq_input %>% layer_gru(64)
  non_seq <- non_seq_input %>% layer_dense(64)
  
  # Concatenate processed inputs
  input <- layer_concatenate(list(seq, non_seq))

  # Define the network structure
  x <- layer_dense(units = 256,
                   activation = "relu")(input)
  x <- x + layer_dense(units = 256,
                       activation = "relu")(x)
  x <- layer_dense(units = 128,
                   activation = "relu")(x)
  x <- x + layer_dense(units = 128,
                       activation = "relu")(x)
  x <- layer_dense(units = 64,
                   activation = "relu")(x)

  # Output layer for predicting error
  err_pred <- layer_dense(name = "error_pred",
                          units = 1)(x)

  # Instantiate and compile models for X, Y, Z coordinates with their respective training data

  stwd(models_directory)

  # Model for X coordinate
  model_X <- keras_model(
    inputs <- list(seq_input, non_seq_input),
    outputs <- err_pred
  )

  model_X %>% compile(
    optimizer = "adam",
    loss =  "mse",
    metrics = "mae"
  )

  model_X %>% fit(
    list(seq = seq_data_train, non_seq = non_seq_data_train),
    y_X_train,
    epochs = 300,
    batch_size = 128,
    callbacks = list(callback_model_checkpoint(paste0(model_name,"_X_",seed,".hdf5"),
                                               monitor = "mae",
                                               mode = "min",
                                               save_best_only = TRUE)
    )
  )

  # Model for Y coordinate

  model_Y <- keras_model(
    inputs <- list(seq_input, non_seq_input),
    outputs <- err_pred
  )

  model_Y %>% compile(
    optimizer = "adam",
    loss =  "mse",
    metrics = "mae"
  )

  model_Y %>% fit(
    list(seq = seq_data_train, non_seq = non_seq_data_train),
    y_Y_train,
    epochs = 300,
    batch_size = 128,
    callbacks = list(callback_model_checkpoint(paste0(model_name,"_Y_",seed,".hdf5"),
                                               monitor = "mae",
                                               mode = "min",
                                               save_best_only = TRUE)
    )
  )

  # Model for Z coordinate
  model_Z <- keras_model(
    inputs <- list(seq_input, non_seq_input),
    outputs <- err_pred
  )

  model_Z %>% compile(
    optimizer = "adam",
    loss =  "mse",
    metrics = "mae"
  )

  model_Z %>% fit(
    list(seq = seq_data_train, non_seq = non_seq_data_train),
    y_Z_train,
    epochs = 300,
    batch_size = 128, 
    callbacks = list(callback_model_checkpoint(paste0(model_name,"_Z_",seed,".hdf5"),
                                               monitor = "mae",
                                               mode = "min",
                                               save_best_only = TRUE)
    )
  )
}
