import os
import urllib.request
import zipfile
import sys

def download_mendeley():
    url = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded"
    zip_path = "ZhangLabData.zip"
    
    print("Downloading Mendeley dataset (this may take a while, it's a few GBs)...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        # Get file size
        total_size = int(response.info().get('Content-Length', 0))
        downloaded = 0
        chunk_size = 8192
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            
            # Print progress
            if total_size > 0:
                percent = int((downloaded / total_size) * 100)
                sys.stdout.write(f"\rProgress: {percent}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                sys.stdout.flush()
                
    print("\nDownload complete! Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete. Deleting ZIP file...")
    os.remove(zip_path)

if __name__ == "__main__":
    download_mendeley()
