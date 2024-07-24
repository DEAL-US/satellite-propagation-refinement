# Load necessary libraries
library(asteRisk)
library(asteRiskData)
library(tidyr)
library(dplyr)

# Function to get orbital parameters from position and velocity vectors.
# pos and vel must be in metres and metres per second, respectively.
get_orbital_parameters <- function(dateTime, pos, vel){
  # Convert dateTime to Modified Julian Date in UTC
  date_mjd_utc <- dateTimeToMJD(dateTime, timeSystem = "UTC")
  # Get quaternion for rotation from GCRF to TEME at the given dateTime
  quaternion <- asteRisk:::rotationGCRFtoTEME(date_mjd_utc)
  # Convert quaternion to Direction Cosine Matrix (DCM)
  dcm <- asteRisk:::quaternionToDCM(quaternion)
  # Rotate position and velocity vectors from GCRF to TEME
  pos <- as.numeric(dcm %*% pos)
  vel <- as.numeric(dcm %*% vel)
  # Convert position and velocity from ECI to Keplerian Orbital Elements (KOE)
  koe <- asteRisk::ECItoKOE(as.vector(pos), as.vector(vel))
  # Calculate orbital parameters from KOE
  n0 <- asteRisk:::semiMajorAxisToMeanMotion(koe$semiMajorAxis)*((2*pi)/(1440))
  e0 <- koe$eccentricity
  i0 <- koe$inclination
  M0 <- koe$meanAnomaly
  omega0 <- koe$argumentPerigee
  OMEGA0 <- koe$longitudeAscendingNode
  # Return the calculated orbital parameters
  return(c(n0, e0, i0, M0, omega0, OMEGA0))
}

# Function to find the closest date within a dataset to a given date within a margin of error
search_date <- function(dataset, date, error_margin_date){
  # Calculate the absolute difference in time between each dataset entry and the given date
  time_differences <- abs(as.POSIXct(dataset$ephemerisUTCTime, tz='UTC')-as.POSIXct(date, tz='UTC'))
  # Find the closest date to the given date
  next_date <- dataset$ephemerisUTCTime[which.min(time_differences)]
  # Calculate the difference in time between the closest date and the given date
  dif_next_date <- difftime(next_date, date, units="secs")
  # Check if the difference is within the margin of error
  if (dif_next_date < error_margin_date) {
    return(next_date)
  } else {
    return("MARGEN EXCEEDED")
  }
}