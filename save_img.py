import csv
import requests
import os
import sys
import time

def put_images(FILE_NAME):
    urls = []
    with open(FILE_NAME, newline="") as csvfile:
        doc = csv.reader(csvfile)
        for row in doc:
            if row and row[0].startswith("https"):
                urls.append(row[0])

    print(f"Total URLs found: {len(urls)}")

    if not os.path.isdir("downloaded_images"):
        os.mkdir("downloaded_images")

    t0 = time.time()
    for idx, url in enumerate(urls):
        print(f"Starting download {idx + 1} of {len(urls)}")
        try:
            resp = requests.get(url, stream=True)
            if resp.status_code == 200:
                image_name = url.split("/")[-1]
                file_path = os.path.join("downloaded_images", image_name)
                with open(file_path, 'wb') as outfile:
                    outfile.write(resp.content)
                
                print(f"Done downloading {idx + 1} of {len(urls)}")
            else:
                print(f"Failed to download URL {idx + 1}. Status code: {resp.status_code}")
        except Exception as e:
            print(f"Failed to download URL {idx + 1}: {str(e)}")
    
    t1 = time.time()
    print(f"Done with download, job took {t1 - t0} seconds")

def main():
    if len(sys.argv) != 2:
        print("Usage: python save_img.py <CSV_FILE_PATH>")
        sys.exit(1)

    FILE_NAME = sys.argv[1]
    put_images(FILE_NAME)

if __name__ == '__main__':
    main()
