library(dplyr)
library(keras)
library(tensorflow)

# Define directories for data and models
data_directory <- "path_to_data_directory"
models_directory <- "path_to_models_directory"
model_name <- "pos"

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
                        # dX_base_real,
                        # dY_base_real,
                        # dZ_base_real,
                        unixTime_predict,
                        # mov_X_sgdp4,
                        # mov_Y_sgdp4,
                        # mov_Z_sgdp4
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

# Read sequential data
seq_data <- as.matrix(read.table("sequential_data_slj.txt"))
n_col <- 9 # 9 for slj and 15 for svlmj


# Loop through each seed value provided in the 'seeds' variable
for (seed in seeds) {
  
  # Set the random seed for TensorFlow operations to ensure reproducibility
  tensorflow::set_random_seed(seed)
  
  # Randomly sample half of the data indices for training
  sample <- sample(nrow(data), as.integer(0.5*nrow(data)))
  
  # Split non-sequential data into training and test sets and convert to tensors
  non_seq_data_train  <- non_seq_data[sample,]
  non_seq_data_train <- as_tensor(non_seq_data_train)
  non_seq_data_test  <- non_seq_data[-sample,]
  non_seq_data_test <- as_tensor(non_seq_data_test)
  
  # Split sequential data into training and test sets, reshape, and convert to tensors
  seq_data_train <- seq_data[sample,]
  seq_data_train <- as_tensor(seq_data_train, shape = c(nrow(seq_data_train), 10, n_col))
  seq_data_test <- seq_data[-sample,]
  seq_data_test <- as_tensor(seq_data_test, shape = c(nrow(seq_data_test), 10, n_col))
  
  # Split target variable into training and test sets and convert to tensors
  y_train <- y[sample,]
  y_train <- as_tensor(y_train)
  y_test <- y[-sample,]
  y_test <- as_tensor(y_test)
  
  # Normalize sequential data by reshaping, calculating IQR and median, and applying normalization
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train)*10, n_col))
  iqr_seq <- apply(seq_data_train, 2, IQR)
  median_seq <- apply(seq_data_train, 2, median)
  seq_data_train <- (seq_data_train-median_seq) / iqr_seq
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train)/10, 10, n_col))
  
  # Normalize non-sequential training data using IQR and median
  iqr_non_seq <- apply(non_seq_data_train, 2, IQR)
  median_non_seq <- apply(non_seq_data_train, 2, median)
  non_seq_data_train <- (non_seq_data_train-median_non_seq) / iqr_non_seq
  
  # Normalize target variable for training data
  iqr_y <- apply(y_train, 2, IQR)
  median_y <- apply(y_train, 2, median)
  y_train <- (y_train-median_y) / iqr_y

  # Normalize test data using training data statistics
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test)*10, n_col))
  seq_data_test <- (seq_data_test-median_seq) / iqr_seq
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test)/10, 10, n_col))
  non_seq_data_test <- (non_seq_data_test-median_non_seq) / iqr_non_seq
  y_test <- (y_test-median_y) / iqr_y

  # Define model inputs for sequential and non-sequential data
  seq_input <- layer_input(shape = c(10, n_col), name = "seq")
  non_seq_input <- layer_input(shape = c(ncol(non_seq_data)), name = "non_seq")

  # Define the model architecture using GRU and dense layers
  seq <- seq_input %>% layer_gru(64)
  non_seq <- non_seq_input %>% layer_dense(64)
  input <- layer_concatenate(list(seq, non_seq))
  x <- layer_dense(units = 256, activation = "relu")(input)
  x <- x + layer_dense(units = 256, activation = "relu")(x)
  x <- layer_dense(units = 128, activation = "relu")(x)
  x <- x + layer_dense(units = 128, activation = "relu")(x)
  x <- layer_dense(units = 64, activation = "relu")(x)
  err_pred <- layer_dense(name = "error_pred", units = 3)(x)

  # Instantiate and compile the model
  model <- keras_model(inputs = list(seq_input, non_seq_input), outputs = err_pred)
  model %>% compile(optimizer = "adam", loss =  "mse", metrics = "mae")

  # Train the model
  setwd(models_directory)
  model %>% fit(
    list(seq = seq_data_train, non_seq = non_seq_data_train),
    y_train,
    epochs = 300,
    batch_size = 128,
    callbacks = list(callback_model_checkpoint(paste0(model_name,"_",seed,".hdf5"),
                                               monitor = "mae",
                                               mode = "min",
                                               save_best_only = TRUE)
    )
  )
}
