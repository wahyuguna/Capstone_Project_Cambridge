import pandas as pd
import folium
from geopy.geocoders import Nominatim



# Read the CSV file
input_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"
output_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\html\point.html"
df = pd.read_csv(input_csv_file, encoding='utf-8')

# membuat visualisasi peta
map1 = folium.Map(
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


for lat, lan, text, sentiment in zip(df['latitude'], df['longitude'], df['text'], df['sentimen']):
    # Marker() takes location coordinates
    # as a list as an argument
    folium.Marker(location=[lat, lan], popup=text,
                  icon=folium.Icon(color=color(sentiment))).add_to(map1)

# Save the HTML file
map1.save(output_file)
print(f"Map saved at {output_file}")
