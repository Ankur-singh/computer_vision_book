# source: https://github.com/ndrplz/google-drive-downloader
import cv2
import numpy as np
import requests
import zipfile
import warnings
from sys import stdout
from os import makedirs
from os.path import dirname
from os.path import exists

def download_from_google_drive(file_id, dest_path, overwrite=False, unzip=False, showsize=False):
    """
    Downloads a shared file from google drive into a given folder.
    Optionally unzips it.

    Parameters
    ----------
    file_id: str
        the file identifier.
        You can obtain it from the sharable link.
    dest_path: str
        the destination where to save the downloaded file.
        Must be a path (for example: './downloaded_file.txt')
    overwrite: bool
        optional, if True forces re-download and overwrite.
    unzip: bool
        optional, if True unzips a file.
        If the file is not a zip file, ignores it.
    showsize: bool
        optional, if True print the current download size.
    Returns
    -------
    None
    """

    CHUNK_SIZE = 32768
    BASE_URL = 'https://docs.google.com/uc?export=download'

    destination_directory = dirname(dest_path)
    if not exists(destination_directory):
        makedirs(destination_directory)

    if not exists(dest_path) or overwrite:

        session = requests.Session()

        print('Downloading {} into {}... '.format(file_id, dest_path), end='')
        stdout.flush()

        response = session.get(BASE_URL, params={'id': file_id}, stream=True)

        token = _get_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(BASE_URL, params=params, stream=True)

        if showsize:
            print()  # Skip to the next line

        current_download_size = [0]
        _save_response_content(response, dest_path, showsize, current_download_size, CHUNK_SIZE)
        print('Done.')

        if unzip:
            try:
                print('Unzipping...', end='')
                stdout.flush()
                with zipfile.ZipFile(dest_path, 'r') as z:
                    z.extractall(destination_directory)
                print('Done.')
            except zipfile.BadZipfile:
                warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination, showsize, current_size, CHUNK_SIZE):
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                if showsize:
                    print('\r' + sizeof_fmt(current_size[0]), end=' ')
                    stdout.flush()
                    current_size[0] += CHUNK_SIZE

# From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Yi', suffix)