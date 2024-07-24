# Load necessary libraries
library(asteRisk)
library(asteRiskData)

# Read the satellite data
datos <- read.table("dataset.txt", header = TRUE)
# Read celestial body data
datos_astros <- read.table("celestial_data.txt", header = TRUE)

# Initialize an empty dataframe to store the combined satellite and celestial data
datos_ampliados <- data.frame()

# Loop through each row in the satellite data
for (i in 1:nrow(datos)) {
  # Print the current iteration number every 1000 iterations for progress tracking
  if (i%%1000==0) {
    print(i)
  }
  # Extract the current row of satellite data
  datos_satelite <- datos[i,]
  # Find the corresponding celestial data by matching the UTC time
  datos_astros_satelite <- datos_astros[(datos_astros$UTCTime==datos_satelite$ephemerisUTCTime_base), -1]
  # Combine the current row of satellite data with its corresponding celestial data
  new_row <- cbind(datos_satelite, datos_astros_satelite)
  # Append the combined data as a new row to the expanded dataset
  datos_ampliados <- rbind(datos_ampliados, new_row)
}

# Write the expanded dataset to a text file, excluding row names for clarity
write.table(datos_ampliados, file = "dataset_with_celestial_data.txt", row.names = FALSE)