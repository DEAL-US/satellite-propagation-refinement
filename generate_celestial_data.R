library(asteRisk)
library(asteRiskData)

# Read satellite data
data <- read.table("dataset.txt", header=TRUE)

# Initialize an empty dataframe to store celestial data
celestial_data <- data.frame()

# Extract unique dates from the satellite data to avoid duplicate calculations
unique_dates <- data$ephemerisUTCTime_base[!duplicated(data$ephemerisUTCTime_base)]

# Loop through each unique date to calculate celestial positions
for (date in unique_dates) {
  # Convert the date to Modified Julian Date (MJD) in UTC time system
  MJD_UTC <- dateTimeToMJD(date, timeSystem = "UTC")
  # Retrieve ephemerides for the given MJD_UTC for Earth as the central body
  ephemerides <- JPLephemerides(MJD_UTC, timeSystem = "UTC", centralBody="Earth")
  
  # Create a new row with the UTC time, Unix time, and positions of celestial bodies
  new_row <- data.frame("UTCTime" = date,
                        "unixTime" = as.numeric(as.POSIXct(date, tz="UTC")),
                        "X_Sun" = ephemerides$positionSun[1],
                        "Y_Sun" = ephemerides$positionSun[2],
                        "Z_Sun" = ephemerides$positionSun[3],
                        "X_Mercury" = ephemerides$positionMercury[1],
                        "Y_Mercury" = ephemerides$positionMercury[2],
                        "Z_Mercury" = ephemerides$positionMercury[3],
                        "X_Venus" = ephemerides$positionVenus[1],
                        "Y_Venus" = ephemerides$positionVenus[2],
                        "Z_Venus" = ephemerides$positionVenus[3],
                        "X_Mars" = ephemerides$positionMars[1],
                        "Y_Mars" = ephemerides$positionMars[2],
                        "Z_Mars" = ephemerides$positionMars[3],
                        "X_Jupiter" = ephemerides$positionJupiter[1],
                        "Y_Jupiter" = ephemerides$positionJupiter[2],
                        "Z_Jupiter" = ephemerides$positionJupiter[3],
                        "X_Saturn" = ephemerides$positionSaturn[1],
                        "Y_Saturn" = ephemerides$positionSaturn[2],
                        "Z_Saturn" = ephemerides$positionSaturn[3],
                        "X_Uranus" = ephemerides$positionUranus[1],
                        "Y_Uranus" = ephemerides$positionUranus[2],
                        "Z_Uranus" = ephemerides$positionUranus[3],
                        "X_Neptune" = ephemerides$positionNeptune[1],
                        "Y_Neptune" = ephemerides$positionNeptune[2],
                        "Z_Neptune" = ephemerides$positionNeptune[3],
                        "X_Pluto" = ephemerides$positionPluto[1],
                        "Y_Pluto" = ephemerides$positionPluto[2],
                        "Z_Pluto" = ephemerides$positionPluto[3],
                        "X_Moon" = ephemerides$positionMoon[1],
                        "Y_Moon" = ephemerides$positionMoon[2],
                        "Z_Moon" = ephemerides$positionMoon[3],
                        "lunar_libration_Phi" = ephemerides$lunarLibrationAngles[1],
                        "lunar_libration_Theta" = ephemerides$lunarLibrationAngles[2],
                        "lunar_libration_Psi" = ephemerides$lunarLibrationAngles[3]
  )
  # Append the new row to the celestial_data dataframe
  celestial_data <- rbind(celestial_data, new_row)
}

# Save the celestial dataset
write.table(celestial_data, file = "celestial_data.txt", row.names = FALSE)