import pandas as pd
import pyreadstat
import os

# Directory containing the .xpt files
data_dir = "data/raw_data/"  # Replace with your directory path

# List all .xpt files in the directory
xpt_files = [file for file in os.listdir(data_dir) if file.endswith(".xpt")]

# Initialize an empty DataFrame
merged_df = None

# Iterate through each .xpt file and merge on SEQN
for file in xpt_files:
    file_path = os.path.join(data_dir, file)
    
    # Read .xpt file
    df = pd.read_sas(file_path)
    
    # Merge with the existing DataFrame
    if merged_df is None:
        merged_df = df  # Initialize with the first DataFrame
    else:
        merged_df = pd.merge(merged_df, df, on="SEQN", how="outer")

# Save the merged DataFrame to a CSV file
merged_df.to_csv("data/dataset.csv", index=False)
