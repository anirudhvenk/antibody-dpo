import pandas as pd

file_path = "data/dpo_data.csv"

# Load the dataset
df_full = pd.read_csv(file_path)

# Sort by similarity in descending order
df_sorted_full = df_full.sort_values(by="similarity", ascending=False)

# Extract the top 100k rows
df_top_100k = df_sorted_full.head(100000)

# Save the top 100k rows to a new CSV file
output_file = "data/dpo/top_100k.csv"
df_top_100k.to_csv(output_file, index=False)