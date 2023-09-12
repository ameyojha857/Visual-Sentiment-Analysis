import requests
import pandas as pd
import sys
import xml.etree.ElementTree as ET
key = 'aa9cbb0147c5a510fbb0a4f179dacbce'
def get_urls(image_tag, MAX_COUNT):
    urls = []
    page = 1
    per_page = 50

    while len(urls) < MAX_COUNT:
        url = f'https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key={key}&text={image_tag}&per_page={per_page}&page={page}&extras=url_o&format=rest'

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data from Flickr API. Status code: {response.status_code}")
            return

        root = ET.fromstring(response.text)

        for photo in root.findall('.//photo'):
            if len(urls) < MAX_COUNT:
                url = photo.get('url_o')
                if url:
                    urls.append(url)
                else:
                    print(f"Url for image number {len(urls) + 1} could not be fetched")
            else:
                print(f"Done fetching {MAX_COUNT} urls")
                break

        page += 1

    urls = pd.Series(urls)
    print(f"Writing out {len(urls)} urls in the current directory")
    urls.to_csv(f"{image_tag}_urls.csv", index=False)
    print("Done!!!")

def main():
    if len(sys.argv) != 3:
        print("Usage: python gen_img.py <image_tag> <MAX_COUNT>")
        return

    tag = sys.argv[1]
    MAX_COUNT = int(sys.argv[2])
    get_urls(tag, MAX_COUNT)

if __name__ == '__main__':
    main()
