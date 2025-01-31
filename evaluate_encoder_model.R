library(dplyr)
library(keras)
library(tensorflow)

source("custom_layers.R")

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
seq_data <- as.matrix(read.table("sequential_data_smj.txt"))

num_timesteps <- 10
num_objects <- 3 # 3 for smj and 5 for svmmj
num_coordinates <- 3
dim_emb_pos <- 16
dim_emb_astros <- 8

a <- read.table(file = "error_models.txt", header = TRUE)

# Loop through each seed value provided in the 'seeds' variable
for (seed in seeds) {
  
  print(seed)
  
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
  seq_data_train <- as_tensor(seq_data_train, shape = c(nrow(seq_data_train), 10, num_objects*num_coordinates))
  seq_data_test <- seq_data[-sample,]
  seq_data_test <- as_tensor(seq_data_test, shape = c(nrow(seq_data_test), 10, num_objects*num_coordinates))
  
  # Split target variable into training and test sets and convert to tensors
  y_train <- y[sample,]
  y_train <- as_tensor(y_train)
  y_test <- y[-sample,]
  y_test <- as_tensor(y_test)
  
  # Normalize sequential data by reshaping, calculating IQR and median, and applying normalization
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train)*10, num_objects*num_coordinates))
  iqr_seq <- apply(seq_data_train, 2, IQR)
  median_seq <- apply(seq_data_train, 2, median)
  seq_data_train <- (seq_data_train-median_seq) / iqr_seq
  seq_data_train <- tf$reshape(seq_data_train, shape = shape(nrow(seq_data_train)/10, 10, num_objects*num_coordinates))
  
  # Normalize non-sequential training data using IQR and median
  iqr_non_seq <- apply(non_seq_data_train, 2, IQR)
  median_non_seq <- apply(non_seq_data_train, 2, median)
  non_seq_data_train <- (non_seq_data_train-median_non_seq) / iqr_non_seq
  
  # Normalize target variable for training data
  iqr_y <- apply(y_train, 2, IQR)
  median_y <- apply(y_train, 2, median)
  y_train <- (y_train-median_y) / iqr_y
  
  # Normalize test data using training data statistics
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test)*10, num_objects*num_coordinates))
  seq_data_test <- (seq_data_test-median_seq) / iqr_seq
  seq_data_test <- tf$reshape(seq_data_test, shape = shape(nrow(seq_data_test)/10, 10, num_objects*num_coordinates))
  non_seq_data_test <- (non_seq_data_test-median_non_seq) / iqr_non_seq
  y_test <- (y_test-median_y) / iqr_y
  
  
  # Reshape seq_data_test to match the expected input shape (batch_size, sequence_length, features)
  seq_data_test <- tf$stack(lapply(1:num_objects, function(i) {
    # Calculate the starting column index for the current object
    start_col <- as.integer((i - 1) * as.integer(num_coordinates))
    
    # Extract a slice from the sequence training data
    tf$slice(seq_data_train,
             begin = list(0L, 0L, start_col),  # Start at Batch=0, Time=0, and the computed Column index
             size = list(as.integer(tf$shape(seq_data_train)[[0]]),
                         as.integer(num_timesteps),
                         as.integer(num_coordinates)))
  }), axis = 1)
  
  
  
  # Define model inputs for sequential and non-sequential data
  seq_input <- layer_input(shape = c(num_objects, num_timesteps, num_coordinates), name = "seq") # (B,num_objects,10,3)
  non_seq_input <- layer_input(shape = c(ncol(non_seq_data)), name = "non_seq")
  
  
  # Positional embeddings
  positional_embedding <- positional_embedding_layer(
    num_timesteps = num_timesteps,
    num_objects = num_objects,
    dim_emb = dim_emb_pos
  )
  embedded_position_sequence <- positional_embedding(seq_input)
  
  
  # Celestial body embedding
  celestial_body_embedding <- celestial_body_embedding_layer(
    num_astros = num_objects,
    astro_embedding_dim = dim_emb_astros
  )
  celestial_body_embedding_sequence <- celestial_body_embedding(embedded_position_sequence) # (B, num_objects, num_timesteps, num_coordinates+dim_emb_pos+dim_emb_astros)
  
  # Flatten to match (B, T, dim), which is the input size for layer_multi_head_attention 
  flatten_sequence <- flatten_astros(celestial_body_embedding_sequence)
  
  
  # Encoder 
  aux <- layer_multi_head_attention(num_heads = 3,
                                    key_dim = 16)
  seq_att <- aux(flatten_sequence, flatten_sequence)
  
  seq_add_1 <- layer_add()(
    list(flatten_sequence, seq_att)
  )
  seq_norm_1 <- layer_normalization()(seq_add_1)
  
  d_model <- num_coordinates+dim_emb_astros+dim_emb_pos
  d_ff <- d_model/2
  seq_ff <- layer_dense(units = d_ff)(seq_norm_1)
  
  seq_ff <- layer_dense(units = d_model, activation = "relu")(seq_ff)
  
  seq_add_2 <- layer_add()(
    list(seq_norm_1, seq_ff)
  )
  seq_norm_2 <- layer_normalization()(seq_add_2)
  
  
  # Gru layer
  seq_gru <- seq_norm_2 %>% layer_gru(256)
  
  
  non_seq <- non_seq_input %>% layer_dense(64)
  concat <- layer_concatenate(list(seq_gru, non_seq))
  x <- layer_dense(units = 256, activation = "relu")(concat)
  x <- x + layer_dense(units = 256, activation = "relu")(x)
  x <- layer_dense(units = 128, activation = "relu")(x)
  x <- x + layer_dense(units = 128, activation = "relu")(x)
  x <- layer_dense(units = 64, activation = "relu")(x)
  err_pred <- layer_dense(name = "error_pred", units = 3)(x)
  
  # Instantiate and compile the model
  model <- keras_model(inputs = list(seq_input, non_seq_input), outputs = err_pred)
  model %>% compile(optimizer = "adam", loss = "mse", metrics = "mae")
  
  
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
  
  # Load and evaluate model
  setwd(models_directory)
  model <- load_model_weights_hdf5(model, filepath=paste0(model_name,"_",seed,".h5"))
  
  result <- model %>% evaluate(list(seq = seq_data_test, non_seq = non_seq_data_test),
                               y_test
  )
  
  error_predictions_normalized <- model %>% predict(list(seq = seq_data_test, non_seq = non_seq_data_test))
  
  error_predictions <- as_tensor(as.matrix(unname(error_predictions_normalized)),
                                 shape = shape(nrow(error_predictions_normalized), 3)
  )
  error_predictions <- error_predictions * iqr_y + median_y
  
  non_corrected_predictions <- select(d_test,
                                      X_predict_sgdp4,
                                      Y_predict_sgdp4,
                                      Z_predict_sgdp4)
  non_corrected_predictions <- as.matrix(unname(non_corrected_predictions))
  non_corrected_predictions <- as_tensor(non_corrected_predictions,
                                         shape = shape(nrow(non_corrected_predictions), 3))
  
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
  a <- rbind(a, b)
  
}
  
setwd(data_directory)
write.table(c, file = "error_models.txt", row.names = FALSE, col.names = TRUE)