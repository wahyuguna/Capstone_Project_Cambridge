import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Initialize Nominatim geocoder
locator = Nominatim(user_agent="OJK-Indonesia")

# Define file paths
input_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\processed_data.csv"
output_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"

# Read the input CSV file
df = pd.read_csv(input_csv_file, encoding='utf-8')

# Function to geocode the location with retries and delays
def geocode_with_retries(location):
    try:
        geocode = RateLimiter(locator.geocode, min_delay_seconds=2, max_retries=5, error_wait_seconds=10)
        return geocode(location)
    except Exception as e:
        print(f"Error geocoding {location}: {e}")
        return None

# Apply geocoding and create 'address' column
print("Geocoding locations...")
df['address'] = df['location'].apply(geocode_with_retries)

# Create 'point' column from geocoded address
df['point'] = df['address'].apply(lambda loc: tuple(loc.point) if loc else None)

# Save to CSV
df.to_csv(output_csv_file, index=False)
print(f"Geocoding completed. Results saved to {output_csv_file}")