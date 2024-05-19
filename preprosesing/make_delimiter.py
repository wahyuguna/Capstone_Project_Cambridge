import pandas as pd

# Path to the Excel file
excel_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\dataset_merge.xlsx"

# Read Excel file
df = pd.read_excel(excel_file, sheet_name='dataset_merge')

# Print the first few rows of the DataFrame to verify the data
print(df.head())

# Path and filename for the CSV file
output_csv_file = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\processed_data.csv"

# Save DataFrame to CSV with a specific delimiter (e.g., semicolon)
df.to_csv(output_csv_file, index=False, sep=',')

print(f"DataFrame has been saved to {output_csv_file}")