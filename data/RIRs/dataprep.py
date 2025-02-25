import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
import tarfile

from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
from os.path import join as opj

parser = argparse.ArgumentParser(description = "MUSAN noise dataset downloader");
parser.add_argument('--save_path', type=str, default=".", help='Target directory');

parser.add_argument('--download', dest='download', action='store_true', help='Enable download')
parser.add_argument('--extract',  dest='extract',  action='store_true', help='Enable extract')
parser.add_argument('--convert',  dest='convert',  action='store_true', help='Enable convert')


args = parser.parse_args();



## ========== ===========
## Download with wget
## ========== ===========
def download(args):
    
    url = 'https://www.openslr.org/resources/28/rirs_noises.zip'
    outfile = url.split('/')[-1]
    
    # Download file
    out = subprocess.call('wget %s -O %s/%s' % (url, args.save_path, outfile), shell=True)
    if out != 0:
        raise ValueError('Download failed %s.'%url)


## ========== ===========
## Extract zip files
## ========== ===========
def full_extract(args, fname):
    
    print('Extracting %s'%fname)
    # if fname.endswith(".tar.gz"):
    #     with tarfile.open(fname, "r:gz") as tar:
    #         safe_extract(tar, args.save_path)
    # elif fname.endswith(".zip"):
    #     with ZipFile(fname, 'r') as zf:
    #         zf.extractall(args.save_path)            
    with ZipFile(fname, 'r') as zf:
        zf.extractall(args.save_path)            


if __name__ == "__main__":
    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    if args.download:
        download(args)
        
    if args.extract:
        full_extract(args, opj(args.save_path, 'rirs_noises.zip'))
