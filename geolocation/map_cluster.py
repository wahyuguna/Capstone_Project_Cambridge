import multiprocessing as mp
from multiprocessing import Pool
import csv
import requests
import string
import pandas as pd
import numpy as np
import geopy
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster


from geopy.geocoders import Nominatim
locator = Nominatim(user_agent="myGeocoder")
from geopy.extra.rate_limiter import RateLimiter


# Read the CSV file
input_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"
output_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\html\map_cluster.html"
df = pd.read_csv(input_csv_file, encoding='utf-8')

# membuat visualisasi peta
map = folium.Map(
    # mengatur latitude dan longitude
    location=[-2.548926, 118.0148634],
    # mengatur zoom pada peta
    zoom_start=5,
)


def color(sentiment):
    if sentiment == 'Positif':
        col = 'green'
    elif sentiment == 'Negatif':
        col = 'red'
    else:
        col = 'blue'  # Handle neutral or other sentiment values
    return col


marker_cluster = MarkerCluster().add_to(map)

for lat, lan, text, sentiment in zip(df['latitude'], df['longitude'], df['text'], df['sentimen']):
    # Marker() takes location coordinates
    # as a list as an argument
    folium.CircleMarker(location=[lat, lan],
                        radius=9,
                        popup=text,
                        fill_color=color(sentiment),
                        color="gray",
                        fill_opacity=0.9).add_to(marker_cluster)


map.save(output_file)
print(f"Map saved at {output_file}")