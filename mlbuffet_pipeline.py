# import kfp
# import kfp.components as comp
import zipfile

import requests

# Constants and parameters
remote_zip = 'https://github.com/AlfonsoGonz/kubeflow-mlbuffet-pipeline/raw/main/files/files.zip'
tag = 'lstm'
model = 'LSTM2FEATURES.pb'
mlbuffet_url = f'http://192.168.1.112:30313/api/v1/train/{tag}/{model}'


def download_and_unzip(file_url):
    response = requests.get(file_url)

    with open('./files.zip', mode='wb') as file:
        file.write(response.content)

    with zipfile.ZipFile(file='./files.zip', mode='r') as zip_file:
        zip_file.extractall()


def start_training(server_url):
    with open('dataset.csv', 'rb') as dataset, open('./train.py', 'rb') as script, open('requirements.txt',
                                                                                        'rb') as requirements:
        response = requests.post(server_url, files={'dataset': dataset.read(),
                                                    'requirements': requirements.read(),
                                                    'script': script.read()})
    print(response.content)


if __name__ == '__main__':
    download_and_unzip(file_url=remote_zip)
    start_training(server_url=mlbuffet_url)
