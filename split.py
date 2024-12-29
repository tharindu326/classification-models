import os
import shutil
import random
import argparse

"""
USAGE:
python split.py -i TLDataset/ALL/PKG-C-NMC2019/train -v TLDataset/ALL/PKG-C-NMC2019/val -r 0.12

"""

# Function to split data
def split_data(source_dir, dest_dir, split_ratio):
    os.makedirs(dest_dir, exist_ok=True)
    files = os.listdir(source_dir)
    random.shuffle(files)
    split_count = int(len(files) * split_ratio)

    for file in files[:split_count]:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

parser = argparse.ArgumentParser(description='Split dataset into train and validation sets.')
parser.add_argument('-i', '--input', required=True, help='Input dataset directory')
parser.add_argument('-v', '--val', required=True, help='Validation dataset directory')
parser.add_argument('-r', '--ratio', type=float, default=0.12, help='Split ratio for validation set (default: 0.12)')
args = parser.parse_args()

input_cancer = os.path.join(args.input, 'CANCER')
input_no_cancer = os.path.join(args.input, 'NO_CANCER')
val_cancer = os.path.join(args.val, 'CANCER')
val_no_cancer = os.path.join(args.val, 'NO_CANCER')

split_data(input_cancer, val_cancer, args.ratio)
split_data(input_no_cancer, val_no_cancer, args.ratio)

print("Data splitting complete!")
