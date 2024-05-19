import nltk
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os

factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

df_pd = pd.read_csv("lokasi.csv",encoding = 'utf-8')
print(df_pd.columns)
# Pilih kolom yang diinginkan
kolom_yang_diinginkan = ['username', 'user_id','location','address','latitude','longitude']
for kolom in kolom_yang_diinginkan:
    if kolom not in df_pd.columns:
        print(f"Kolom '{kolom}' tidak ditemukan dalam DataFrame.")

# Pilih kolom yang ada
df_baru = df_pd[kolom_yang_diinginkan]

print(df_baru)

# Tentukan path dan nama file untuk menyimpan CSV
file_path = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset"
os.makedirs(file_path, exist_ok=True)
output_filename = os.path.join(file_path, "filtered_data.csv")

# Simpan DataFrame ke file CSV
df_baru.to_csv(output_filename, index=False)

print(f"DataFrame telah disimpan ke {output_filename}")
