library(tensorflow)
library(keras)

# Function to create sinusoidal positional embeddings
create_positional_embeddings <- function(seq_len, d_model) {
  # Create position and dimension indices
  position <- 0:(seq_len-1)
  dimension <- 0:(d_model-1)
  
  # Create matrices for position and dimension values
  pos_matrix <- matrix(position, nrow = seq_len, ncol = d_model)
  dim_matrix <- matrix(dimension, nrow = seq_len, ncol = d_model, byrow = TRUE)
  
  # Compute sinusoidal and cosine positional encoding
  pos_embeddings <- ifelse(dim_matrix %% 2 == 0, 
                           sin(pos_matrix / 10000^(dim_matrix / d_model)),
                           cos(pos_matrix / 10000^((dim_matrix - 1) / d_model)))
  
  # Convert to TensorFlow tensor
  pos_embeddings <- tf$cast(pos_embeddings, dtype = "float32")
  pos_embeddings <- as_tensor(pos_embeddings)
  
  return(pos_embeddings)
}

# Positional Embedding Layer
positional_embedding_layer <- keras::new_layer_class(
  classname = "PositionalEmbeddingLayer",
  
  initialize = function(num_timesteps, num_objects, dim_emb) {
    super()$`__init__`()
    self$num_timesteps <- as.integer(num_timesteps)
    self$num_objects <- as.integer(num_objects)
    self$dim_emb <- as.integer(dim_emb)
  },
  
  call = function(inputs, mask = NULL) {
    if (self$dim_emb == 0) return(inputs) # If embedding dimension is zero, return inputs unchanged
    
    pos_emb <- create_positional_embeddings(self$num_timesteps, self$dim_emb) # (num_timesteps, dim_emb)
    
    # Expand dimensions for broadcasting
    pos_emb <- k_expand_dims(pos_emb, axis = 1) # (1, num_timesteps, dim_emb)
    pos_emb <- k_expand_dims(pos_emb, axis = 1) # (1, 1, num_timesteps, dim_emb)
    
    # Tile (repeat) the positional embeddings to match the input shape
    pos_emb_aux <- tf$tile(pos_emb, multiples = c(tf$shape(inputs)[[0]], self$num_objects, 1L, 1L))
    
    # Concatenate input tensor with positional embeddings along the last dimension
    concatenated <- tf$concat(list(inputs, pos_emb_aux), axis = -1L)
    return(concatenated)
  }
)

# Astro Embedding Layer
celestial_body_embedding_layer <- keras::new_layer_class(
  classname = "AstroEmbeddingLayer",
  
  initialize = function(num_astros, astro_embedding_dim) {
    super()$`__init__`()
    self$num_astros <- as.integer(num_astros)
    self$astro_embedding_dim <- as.integer(astro_embedding_dim)
  },
  
  build = function(input_shape) {
    # Define a trainable matrix for astro embeddings
    self$entrenable_matrix <- self$add_weight(
      name = "entrenable_matrix",
      shape = c(self$num_astros, 1L, self$astro_embedding_dim), #(num_celestial_bodies, 1, emb_celestial_body_dim)
      initializer = "random_normal",
      trainable = TRUE # Make it trainable
    )
  },
  
  call = function(inputs, ...) {
    # Broadcast the trainable astro embedding matrix to match input shape
    astros_emb <- tf$broadcast_to(
      self$entrenable_matrix,
      c(tf$shape(inputs)[[0]], self$num_astros, tf$shape(inputs)[[2]], self$astro_embedding_dim)
      #(B, num_celestial_bodies, num_timesteps, celestial_body_dim)
    )
    
    # Concatenate input tensor with astro embeddings along the last dimension
    concatenated <- tf$concat(list(inputs, astros_emb), axis = -1L)
    return(concatenated)
  }
)


flatten_celestial_body <- layer_lambda(
  f = function(inputs) {
    tf$reshape(inputs,
               shape = c(tf$shape(inputs)[[0]],
                         tf$shape(inputs)[[1]]*tf$shape(inputs)[[2]],
                         tf$shape(inputs)[[3]]
               )
    )
    # (B, num_objects*num_timesteps, num_coordinates + emb_pos_dim + emb_celestial_body_dim)
  }
)


