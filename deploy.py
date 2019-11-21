import os
import tarfile

import requests
from tqdm import tqdm
import sys

def download(url, filename):
    with open(filename, "wb") as f:
        print("Downloading", filename)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            bar = tqdm(total=int(total_length))
            for data in response.iter_content(chunk_size=4096):
                bar.update(len(data))
                f.write(data)


if __name__ == '__main__':
    print('Paste link for models file download')
    link = sys.stdin.readline()
    download(link, 'models.tar.gz')
    tar = tarfile.open('models.tar.gz', "r:gz")
    tar.extractall()
    tar.close()

    os.remove('models.tar.gz')