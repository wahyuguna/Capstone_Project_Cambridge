import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests

# Read CSV file containing usernames
input_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"
output_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"

# Read the input CSV file
df_pd = pd.read_csv(input_csv_file, encoding='utf-8')

# Initialize target URL
base_url = 'https://twitter.com/'

# List to store locations
locations = []

# Loop through unique usernames
for user in df_pd['username'].unique():
    url = base_url + user
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    # Find location element
    loc_elem = soup.find('span', class_='ProfileHeaderCard-locationText')

    # Check if location element exists
    if loc_elem:
        location = loc_elem.get_text(strip=True)
    else:
        location = None

    # Append location to list
    locations.append(location)

# Add location column to DataFrame
df_pd['location'] = locations

# Save DataFrame to CSV
df_pd.to_csv(output_csv_file, index=False)