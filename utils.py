import os
import shutil

# Define the source directory and the destination directory
source_dir = 'dataset'
destination_dir = 'dataset_100'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through each folder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # Ensure that it is a directory
    if os.path.isdir(folder_path):
        # Create a corresponding folder in the destination directory
        new_folder_path = os.path.join(destination_dir, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Copy the first 100 images from each folder to the corresponding new folder
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image in images[:100]:
            shutil.copy(os.path.join(folder_path, image), new_folder_path)

# Print a completion message
print("Dataset versioning completed successfully.")
