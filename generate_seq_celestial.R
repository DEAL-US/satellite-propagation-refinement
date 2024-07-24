library(asteRisk)
library(asteRiskData)

# Read the dataset
data <- read.table("dataset", header=TRUE)

# Define the number of intervals (jumps) between the base and prediction dates
num_jumps <- 9
# Define the number of columns for celestial body features
ncol <- 15

# Extract the base and prediction UTC times from the dataset
date_tuples <- data[c("ephemerisUTCTime_base", "ephemerisUTCTime_predict")]

# Initialize a matrix to store the calculated celestial positions, filled with NA values initially
aux <- matrix(data=NA, nrow=nrow(data), ncol*(num_jumps+1))

# Loop through each row in the dataset
for (i in 1:nrow(data)) {
  
  # Print the current iteration number every 1000 iterations to monitor progress
  if (i%%1000==0) {
    print(i)
  }
  
  # Extract the base and prediction dates for the current row
  tuple <- date_tuples[i,]
  base_date <- tuple$ephemerisUTCTime_base
  prediction_date <- tuple$ephemerisUTCTime_predict
  
  # Calculate the time interval (jump) between each measurement, in seconds
  jump <- difftime(as.POSIXct(prediction_date, tz = "UTC"),
                   as.POSIXct(base_date, tz = "UTC"),
                   units = "secs") / num_jumps
  jump <- as.numeric(jump)
  jump <- round(jump, 4)
  
  # Loop through each jump to calculate and store celestial positions
  for (j in 0:num_jumps) {
    date <- as.POSIXct(base_date, tz = "UTC") + jump*j
    MJD_UTC <- dateTimeToMJD(date, timeSystem = "UTC")
    ephemerides <- JPLephemerides(MJD_UTC, timeSystem = "UTC", centralBody="Earth")
    
    # Store the positions of the Sun, Venus, Moon, Mars, and Jupiter for each jump
    aux[i,(j*ncol+1):((j+1)*ncol)] <- c(ephemerides$positionSun[1],
                                        ephemerides$positionSun[2],
                                        ephemerides$positionSun[3],
                                        ephemerides$positionVenus[1],
                                        ephemerides$positionVenus[2],
                                        ephemerides$positionVenus[3],
                                        ephemerides$positionMoon[1],
                                        ephemerides$positionMoon[2],
                                        ephemerides$positionMoon[3],
                                        ephemerides$positionMars[1],
                                        ephemerides$positionMars[2],
                                        ephemerides$positionMars[3],
                                        ephemerides$positionJupiter[1],
                                        ephemerides$positionJupiter[2],
                                        ephemerides$positionJupiter[3])
  }
}

# Save the matrix of celestial positions to a file, without row or column names
write.table(aux, file = "sequential_data_svmmj.txt", row.names = FALSE, col.names = FALSE)