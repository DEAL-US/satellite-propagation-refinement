library(asteRisk)

# Function to extract GLONASS satellite data from RINEX files
get_datos_GLONASS <- function(ruta_base, ficheros, codigo_satelite){
  # Define a list of days, months, and years to iterate over
  dias <- c("01", "02", "03", "04", "05", "06", "07", "08", "09", as.character(10:31))
  meses <- c("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")
  anos <- c("2019", "2020", "2021", "2022")
  # Initialize an empty vector to store file names
  ficheros <- c()
  # Generate file names for each combination of day, month, and year
  for(d in dias){
    for(m in meses){
      for(a in anos){
        f <- paste(d, m, a, sep="-") # Combine day, month, year with hyphens
        f <- paste0(f, ".g") # Append the file extension
        ficheros <- c(ficheros, f) # Add the file name to the list
      }
    }
  }
  # Initialize an empty dataframe to store satellite data
  dataset <- data.frame("X" = numeric(),
                        "Y" = numeric(),
                        "Z" = numeric(),
                        "dx" = numeric(),
                        "dy" = numeric(),
                        "dz" = numeric(),
                        "ephemerisUTCTime" = character())
  # Iterate over each file name in the list
  for (f in ficheros){
    print(f) # Print the current file name
    faux <- paste0(ruta_base, f) # Create the full file path
    if (file.exists(faux)) { # Check if the file exists
      rinexData <- readGLONASSNavigationRINEX(faux) # Read the RINEX file
      for (i in 1:length(rinexData$messages)) { # Iterate over each message in the RINEX data
        satelliteData <- rinexData$messages[[i]] # Extract satellite data
        if (satelliteData$satelliteNumber==codigo_satelite){ # Filter for the specified satellite number
          # Extract position and velocity data
          position_ITRF <- c(satelliteData$positionX, satelliteData$positionY, satelliteData$positionZ)
          velocity_ITRF <- c(satelliteData$velocityX, satelliteData$velocityY, satelliteData$velocityZ)
          epoch <- as.character.Date(as.POSIXct(satelliteData$ephemerisUTCTime, tz="UTC")) # Convert ephemeris time to character
          coordinates_GCRF <- ITRFtoGCRF(position_ITRF, velocity_ITRF, epoch) # Convert coordinates from ITRF to GCRF
          # Create a new row with the converted data
          new_row <- data.frame("X" = coordinates_GCRF$position[1]*1000,
                                "Y" = coordinates_GCRF$position[2]*1000,
                                "Z" = coordinates_GCRF$position[3]*1000,
                                "dX" = coordinates_GCRF$velocity[1]*1000,
                                "dY" = coordinates_GCRF$velocity[2]*1000,
                                "dZ" = coordinates_GCRF$velocity[3]*1000,
                                "ephemerisUTCTime" = as.character.Date(as.POSIXct(satelliteData$ephemerisUTCTime, tz="UTC")))
          dataset <- rbind(dataset, new_row) # Append the new row to the dataset
        }
      }
    } else {
      print("File does not exist") # Print a message if the file does not exist
    }
  }
  return(dataset) # Return the final dataset
}

# Call the function to extract data for satellite number 17 (Kosmos 2514)
dataset <- get_datos_GLONASS(ruta_base, ficheros, 17)

# Remove duplicate rows from the dataset
dataset <- dataset[!duplicated(dataset),]
anyDuplicated(dataset) # Check for duplicates in the dataset

# Write the dataset to a text file
write.table(dataset, file = "dataset.txt", row.names = FALSE)