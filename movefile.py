import shutil
import os
import glob
import random

# Path to the source folder
dst_folder = './Dataset/Train/John'

# Path to the destination folder
src_folder = './Dataset/Val/John'

# Check if the destination folder exists; if not, create it
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# Find all .mp4 files in the source folder
mp4_files = glob.glob(os.path.join(src_folder, '*.mp4'))

# Shuffle the list of files to randomize the selection
random.shuffle(mp4_files)

# Move the first 45 .mp4 files
for i, file_path in enumerate(mp4_files[:15]):
    # Construct the destination path
    dst_path = os.path.join(dst_folder, os.path.basename(file_path))
    
    # Move the file
    shutil.move(file_path, dst_path)
    print(f"Moved file {i+1}: {file_path} to {dst_path}")

print("All files have been moved.")
