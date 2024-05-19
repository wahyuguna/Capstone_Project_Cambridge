import requests
import pandas as pd
import os


file_path = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset"
search_terms = ['asuransi bumiputra', 'wanaartha life', 'kresna life']

# Buat directory jika belum ada
os.makedirs(file_path, exist_ok=True)

for sch in search_terms:
    print(sch)
    payload = { 'api_key': 'efc55cba6ae945e96132faaf8c1406c1', 'url': 'https://twitter.com','query': sch, 'num':'100' }
    response = requests.get('https://api.scraperapi.com/structured/twitter/search', params=payload)
    # Pastikan respons berhasil
    if response.status_code == 200:
        data = response.json()
        print(data)
        if 'organic_results' in data:
            print(data['organic_results'])
            df = pd.DataFrame(data['organic_results'])

            # Buat nama file dengan benar
            output_filename = os.path.join(file_path, f"{sch.replace(' ', '_')}.csv")
            df.to_csv(output_filename, index=False)
            print(f"Data telah disimpan ke {output_filename}")
        else:
            print(f"Tidak ada hasil organik ditemukan untuk pencarian: {sch}")
    else:
        print(f"Error: {response.status_code} saat mencoba mengakses API untuk pencarian: {sch}")