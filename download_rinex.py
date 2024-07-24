import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path
from os import rename, remove

# Define the main path where the script is located
main_path = Path(__file__).parent.resolve()

# Define the start and end years for the data collection
start_year = 2019
end_year = 2023

# Define the start and end dates
start_date = date(start_year, 1, 1)
end_date = date(end_year, 1, 1)

# Base URL for the data     
BASE_URL = "https://www.juntadeandalucia.es/institutodeestadisticaycartografia/rinex/CRDB/"

# Function to find the ZIP file link on a given URL
def find_zip_link_in_url(url:str):
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # Loop through all anchor tags to find the link ending with ".zip"
    for link in soup.find_all('a'):
        u = link.get('href')
        if u[-4:] == ".zip":
            return u

# Function to generate a range of dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# Loop through each date in the range
for date in tqdm(daterange(start_date, end_date)):
    day = str(date.day)
    month = str(date.month)

    # Format the day and month to ensure they are two digits
    day = f"0{day}" if len(day) == 1 else day
    month = f"0{month}" if len(month) == 1 else month

    # Construct the URL for the specific date
    url = f"{BASE_URL}/{month}/{day}/24H_30seg"

    # Find the ZIP file link
    t = find_zip_link_in_url(url)

    # If no link is found, log the exception and continue to the next date
    if(t is None):
        with open(f"{main_path}/exceptions.txt", "a") as excep_file:
            excep_file.write(f"problem with date: {day}/{month}, url is: {url} \n")
        continue

    # Download the ZIP file
    r = requests.get(t, allow_redirects=True)
    zip_dest = f'./temp/{day}-{month}-{year}.zip'
    open(zip_dest, 'wb').write(r.content)

    # Extract the ZIP file
    with ZipFile(zip_dest, 'r') as zip:
        filelist = zip.namelist()
        for f in filelist:
            # Check the file extension and rename accordingly
            if f[-1:] == "g":
                zip.extract(f, f"{main_path}/downloads")
                rename(f"{main_path}/downloads/{f}", f"{main_path}/downloads/{day}-{month}-{year}.g")

            elif f[-1:] == "n":
                zip.extract(f, f"{main_path}/downloads/")
                rename(f"{main_path}/downloads/{f}", f"{main_path}/downloads/{day}-{month}-{year}.n")
                
    # Remove the ZIP file after extraction
    remove(zip_dest)