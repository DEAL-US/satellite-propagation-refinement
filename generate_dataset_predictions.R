source("utils.R")

# Function to generate predictions for a dataset based on orbital parameters and a given base date
get_df_predictions_from_one_point_sgdp4 <- function(dataset, base_date, times, error_margin, Bstar){
  
  # We suppose base_date is already a date in the dataset,
  # so there's no need to find a close match
  
  # Initialize the dataframe to return
  res_df <- data.frame()
  
  # Get the data for the given base date
  # Select the first row in case of duplicates
  base_data <- dataset[dataset$ephemerisUTCTime==base_date,][1,]
  
  # Calculate orbital parameters from the base data
  orbital_elements <- get_orbital_parameters(dateTime = base_data$ephemerisUTCTime, 
                                             pos = c(base_data$X, base_data$Y, base_data$Z),
                                             vel = c(base_data$dX, base_data$dY, base_data$dZ))
  
  # Initialize an empty vector to store dates for prediction
  prediction_dates <- c()
  for (t in times){
    # Calculate the target date for prediction
    date_to_search <- as.character.Date(as.POSIXct(base_date, tz='UTC')+t)
    date_error_margin <- (1+error_margin)*t
    
    # Find the closest date in the dataset within the error margin
    next_date <- search_date(dataset, date_to_search, date_error_margin)
    
    # If the date is within the margin, add it to the prediction dates
    if (next_date!="MARGEN EXCEEDED") {
      prediction_dates <- c(prediction_dates, next_date)
    }
  }
  
  # Remove duplicate dates from predictions
  prediction_dates <- prediction_dates[!duplicated(prediction_dates)]
  
  # Loop through each date to predict
  for (i in 1:length(prediction_dates)) {
    tryCatch({
      # Calculate the prediction for the current date using the SGDP4 model
      prediction_date <- prediction_dates[i]
      point_TEME <- sgdp4(n0=orbital_elements[1], e0=orbital_elements[2], i0=orbital_elements[3],
                          M0=orbital_elements[4], omega0=orbital_elements[5],
                          OMEGA0=orbital_elements[6], Bstar=Bstar,
                          initialDateTime=base_date, targetTime=prediction_date)
      
      # Convert the prediction from TEME to GCRF coordinate system
      point_GCRF <- TEMEtoGCRF(point_TEME$position*1000, point_TEME$velocity*1000, prediction_date)
    },
    error=function(e){
      # Handle errors by setting prediction to NA
      point_GCRF <- list(NA, NA)
      names(point_GCRF) <- c("position", "velocity")
    },
    warning=function(w){
      # Handle warnings similarly to errors
      point_GCRF <- list(NA, NA)
      names(point_GCRF) <- c("position", "velocity")
    })
    
    # Get the real data for the prediction date
    data_predicted_real <- dataset[dataset$ephemerisUTCTime==prediction_date, ]
    
    # Create a new row with both real and predicted data
    new_row <- data.frame("ephemerisUTCTime_base" = base_date,
                          "unixTime_base" = as.numeric(as.POSIXct(base_date, tz="UTC")),
                          "X_base_real" = base_data$X,
                          "Y_base_real" = base_data$Y,
                          "Z_base_real" = base_data$Z,
                          "dX_base_real" = base_data$dX,
                          "dY_base_real" = base_data$dY,
                          "dZ_base_real" = base_data$dZ,
                          "time_to_predict" = abs(as.POSIXct(base_date, tz='UTC')-as.POSIXct(prediction_date, tz='UTC')),
                          "ephemerisUTCTime_predict" = prediction_date,
                          "unixTime_predict" = as.numeric(as.POSIXct(prediction_date, tz="UTC")),
                          "X_predict_real" = data_predicted_real$X,
                          "Y_predict_real" = data_predicted_real$Y,
                          "Z_predict_real" = data_predicted_real$Z,
                          "dX_predict_real" = data_predicted_real$dX,
                          "dY_predict_real" = data_predicted_real$dY,
                          "dZ_predict_real" = data_predicted_real$dZ,
                          "X_predict_sgdp4" = point_GCRF$position[1],
                          "Y_predict_sgdp4" = point_GCRF$position[2],
                          "Z_predict_sgdp4" = point_GCRF$position[3],
                          "dX_predict_sgdp4" = point_GCRF$velocity[1],
                          "dY_predict_sgdp4" = point_GCRF$velocity[2],
                          "dZ_predict_sgdp4" = point_GCRF$velocity[3]
    )
    
    # Add the new row to the results dataframe
    res_df <- rbind(res_df, new_row)
  }
  # Return the results dataframe
  return(res_df)
}


# Function to generate predictions for multiple base dates using the SGDP4 model
get_df_predictions_sgdp4 <- function(dataset, points_number, times, error_margin, Bstar=0){
  # Parameters:
  # - dataset: The dataset containing orbital data
  # - points_number: The number of random base dates to select for prediction
  # - times: An array of time intervals (in seconds) for which predictions are to be made
  # - error_margin: The acceptable error margin for predictions
  # - Bstar: The drag term of the satellite (default is 0)

  # Select random points from the dataset as base dates for prediction
  random_points <- sample(1:nrow(dataset), points_number, replace=FALSE)
  base_dates <- dataset[random_points,]$ephemerisUTCTime

  # Initialize an empty dataframe to store the results
  res_df <- data.frame()
  
  # Loop through each base date
  for (i in 1:length(base_dates)) {
    # Get the current base date
    date <- base_dates[i]
    # Generate predictions for the current base date
    res_aux <- get_df_predictions_from_one_point_sgdp4(dataset = dataset,
                                                           base_date = date,
                                                           times = times,
                                                           error_margin = error_margin,
                                                           Bstar = Bstar)
    # Combine the predictions with the results dataframe
    res_df <- rbind(res_df, res_aux)
  }
  
  # Return the combined results dataframe
  return(res_df)
}

# Set the number of points (base dates) to use for prediction
points_number <- 40000
# Define target times for prediction in seconds
target_times <- 60 * 60 * c(0.5, 1, 2, 5, 10, 24, 48, 120, 240, 720)
# Set the error margin for predictions
error_margin <- 0.2
# Load the dataset from a text file
dataset <- read.table("dataset.txt", header = TRUE)

# Generate predictions using the defined function and parameters
a <- get_df_predictions_sgdp4(dataset = dataset,
                              points_number = points_number,
                              times = target_times,
                              error_margin = error_margin,
                              Bstar = 0)