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
    
    url = 'https://www.openslr.org/resources/17/musan.tar.gz'
    outfile = url.split('/')[-1]
    
    # Download file
    out = subprocess.call('wget %s -O %s/%s' % (url, args.save_path, outfile), shell=True)
    if out != 0:
        raise ValueError('Download failed %s.'%url)


## ========== ===========
## Extract zip files
## ========== ===========
def is_within_directory(directory, target):
    
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])
    
    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

    for member in tqdm(tar.getmembers()):
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(path, members, numeric_owner=numeric_owner)

def full_extract(args, fname):
    
    print('Extracting %s'%fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            safe_extract(tar, args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(args.save_path)            


## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):

    files = glob.glob('%s/musan/*/*/*.wav'%args.save_path)

    audlen = 16000*5
    audstr = 16000*3

    for idx,file in enumerate(tqdm(files)):
        fs,aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0,len(aud)-audlen,audstr):
            wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])

        print(idx,file)


if __name__ == "__main__":
    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    if args.download:
        download(args)
        
    if args.extract:
        full_extract(args, opj(args.save_path, 'musan.tar.gz'))
        # split_musan(args)
        
    if args.convert:
        split_musan(args)
        
