import zipfile
import os
from tqdm import tqdm
import argparse

"""
USAGE:
python unzip.py -i TLDataset.zip -o TLDataset
"""

# Function to unzip file with progress
def unzip_with_progress(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get the list of files
        file_list = zip_ref.namelist()
        total_files = len(file_list)

        # Extract files with progress bar
        with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)

        print("Extraction complete!")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Unzip a file with progress.')
parser.add_argument('-i', '--input', required=True, help='Input ZIP file path')
parser.add_argument('-o', '--output', required=True, help='Output directory path')
args = parser.parse_args()

# Ensure target directory exists
os.makedirs(args.output, exist_ok=True)

# Run the unzip function
unzip_with_progress(args.input, args.output)
