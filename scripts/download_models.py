import requests
import zipfile
import os
import sys

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = os.path.join(dest_folder, url.split('/')[-1])

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <output_folder>")
        sys.exit(1)

    output_folder = sys.argv[1]
    url = "https://www.dropbox.com/scl/fi/tw6ukedy9oc8anx0s02kk/reni_plus_plus_models.zip?rlkey=smr8ecsw919cokzmot7dksxgx&dl=1"  # dl=1 is important for direct download

    print("Downloading...")
    zip_path = download_file(url, output_folder)

    print("Unzipping...")
    unzip_file(zip_path, output_folder)

    print("Done!")
