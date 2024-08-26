import hashlib
import os
import requests
import zipfile
import tarfile

def download(name, cache_dir=os.path.join('datasets'), data_hub=dict()):
    """
    Download a file inserted into data_hub, return the local filename.
    
    Args:
        name (str): name of the dataset in data_hub
        cache_dir (str): local directory for saving the dataset
        data_hub (dict): global dataset list
    Returns:
        name (str): filename of the dataset
    """
    assert name in data_hub, f"{name} does not exist in {data_hub}."
    url, sha1_hash = data_hub[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """
    Download and extract a zip/tar file.
    
    Args:
        name (str): name of the file to extract
        folder (str): folder where to extract stuffs
    Returns:
        (str): directory where the dataset is extracted
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all(data_hub=dict()):
    """
    Download all files in the data_hub.
    
    Args:
        data_hub (dict): files to extract
    Returns:
        None
    """
    for name in data_hub:
        download(name)
