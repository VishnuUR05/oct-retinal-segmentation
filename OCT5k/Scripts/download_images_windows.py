import os
import zipfile
import shutil
import pandas as pd
import numpy as np
import cv2
import imageio.v2 as imageio

def make_target_dirs(target_paths):
    for dirname in set(os.path.dirname(p) for p in target_paths):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

def download_dataset():
    print("\n--- MANUAL DOWNLOAD STEPS REQUIRED ---")
    print("Mendeley and Google Drive block automated script downloads with Cloudflare/anti-bot protection.")
    print("Please follow these manual steps to get the original images:")
    print("1. Mendeley (ZhangLabData): https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded")
    print("   Extract it so you have 'ZhangLabData'")
    print("2. Heidelberg dataset: https://drive.google.com/u/0/uc?id=1Rv82F7CjPveyONdy1YbRHh05emCb6_Eu&export=download")
    print("   Extract it so you have 'Macular Dataset-Heidelberg'")
    print("3. Rasti dataset: http://cutt.ly/wGOhQeK")
    print("   Extract it using the password from https://sites.google.com/site/hosseinrabbanikhorasgani/")
    print("   Extract those zip files into a folder named 'Macular-Dataset-R.Rasti_old'")
    print("-------------------------------------------\n")

def process_paths(csv_file, is_detection=False, is_automatic=False):
    if not os.path.exists(csv_file):
        print(f"Skipping {csv_file} as it doesn't exist.")
        return
        
    df = pd.read_csv(csv_file, header=None)
    toPath = list(df[0])
    fromPath = list(df[1])
    
    missing_count = 0

    for i in range(len(fromPath)):
        if not os.path.exists(fromPath[i]):
            missing_count += 1
            continue
            
        fromImage = np.array(imageio.imread(fromPath[i]))
        make_target_dirs([toPath[i]])
        
        if is_detection:
            fromImage = cv2.resize(fromImage, dsize=(512, 512))
            toPath[i] = toPath[i].replace(".jpeg", ".png")
            imageio.imsave(toPath[i], fromImage)
        elif is_automatic:
            if ".jpeg" in toPath[i]:
                toPath[i] = toPath[i].replace(".jpeg", ".png")
            fromImage = cv2.resize(fromImage, dsize=(512, 512))
            imageio.imsave(toPath[i], fromImage)
        else:
            # Original and Manual Paths
            if "manual_paths" in csv_file:
                fromImage = cv2.resize(fromImage, dsize=(512, 512))
                imageio.imsave(toPath[i], fromImage)
            else:
                if ".jpeg" in toPath[i]:
                    shutil.copy2(fromPath[i], toPath[i])
                else:
                    imageio.imsave(toPath[i], fromImage)
                    
    print(f"Processed {csv_file}. Missed {missing_count} images because source folders were missing.")

if __name__ == '__main__':
    # Fix paths by always running from the actual OCT5k/OCT5k root folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.dirname(script_dir)
    os.chdir(dataset_root)
    
    print("Starting Windows-compatible dataset download and processing script...")
    
    # 1. Provide instructions
    download_dataset()
    
    # Clean up any bad zip files downloaded previously
    if os.path.exists("ZhangLabData.zip"):
        os.remove("ZhangLabData.zip")
    
    print("\nStep 2: Processing and organizing images...")
    print("Make sure you have manually extracted the Rasti and Heidelberg datasets into this folder before continuing if you want all images.")
    
    # Run the processing steps
    process_paths("Scripts/paths/original_paths.csv")
    process_paths("Scripts/paths/manual_paths.csv")
    process_paths("Scripts/paths/automatic_paths.csv", is_automatic=True)
    process_paths("Scripts/paths/detection_paths.csv", is_detection=True)
    
    print("\nFinished! Run `verify_dataset.py` to check if Images folder is populated correctly.")
