"""
The following scipt is originated from the project https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py
We applied some modifications in distinguishing output folders for dev/test splits, and boosting AAC-to-WAV conversion speed via multiprocessing.
"""
#!/usr/bin/python
#-*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
import tarfile
import pandas as pd

from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
from multiprocessing import Pool, freeze_support

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description = "VoxCeleb downloader");

parser.add_argument('--save_path',    type=str, default="data", help='Target directory');
parser.add_argument('--user',         type=str, default="user", help='Username');
parser.add_argument('--password',     type=str, default="pass", help='Password');
parser.add_argument('--ncpu',         type=int, default=4, help='multi-processing cores for converting AAC files to WAV');
parser.add_argument('--remove_aac',   dest='remove_aac',  action='store_true', help='Remove AAC file right-after the WAV conversion')


parser.add_argument('--download',   dest='download', action='store_true', help='Enable download')
parser.add_argument('--extract',    dest='extract',  action='store_true', help='Enable extract')
parser.add_argument('--convert',    dest='convert',  action='store_true', help='Enable convert')
parser.add_argument('--preprocess', dest='preprocess', action='store_true', help='Enable preprocess')
parser.add_argument('--augment',    dest='augment',  action='store_true', help='Download and extract augmentation files')

args = parser.parse_args();

## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

## ========== ===========
## Download with wget
## ========== ===========
def download(args, lines):

    for line in lines:
        url     = line.split()[0]
        md5gt     = line.split()[1]
        outfile = url.split('/')[-1]

        ## Download files
        out     = subprocess.call('wget %s --user %s --password %s -O %s/%s'%(url,args.user,args.password,args.save_path,outfile), shell=True)
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        ## Check MD5
        md5ck     = md5('%s/%s'%(args.save_path,outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        else:
            raise Warning('Checksum failed %s.'%outfile)

## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args,lines):

    for line in lines:
        infile     = line.split()[0]
        outfile    = line.split()[1]
        md5gt     = line.split()[2]

        ## Concatenate files
        out     = subprocess.call('cat %s/%s > %s/%s' %(args.save_path,infile,args.save_path,outfile), shell=True)

        ## Check MD5
        md5ck     = md5('%s/%s'%(args.save_path,outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        else:
            raise Warning('Checksum failed %s.'%outfile)

        out     = subprocess.call('rm %s/%s' %(args.save_path,infile), shell=True)

## ========== ===========
## Extract zip files
## ========== ===========
def is_within_directory(directory, target):
    
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])
    
    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

    for member in tar.getmembers():
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
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):

    print('Extracting %s'%fname)
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile,args.save_path)
            # pdb.set_trace()
            # zf.extractall(args.save_path)

# ## ========== ===========
# ## Convert: Original script
# ## ========== ===========
# def convert(args):
#
#     files     = glob.glob('%s/voxceleb2/*/*/*.m4a'%args.save_path)
#     files.sort()
#
#     print('Converting files from AAC to WAV')
#     for fname in tqdm(files):
#         outfile = fname.replace('.m4a','.wav')
#         out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fname,outfile), shell=True)
#         if out != 0:
#             raise ValueError('Conversion failed %s.'%fname)

## ========== ===========
## Convert: New script
## ========== ===========
def convert(fpath):
    outfile = fpath.replace('.m4a','.wav')
    out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fpath, outfile), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.'%fpath)
    
    if args.remove_aac:
        out = subprocess.call('rm %s' % (fpath), shell=True)

## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):

    files = glob.glob('%s/musan/*/*/*.wav'%args.save_path)

    audlen = 16000*5
    audstr = 16000*3

    for idx,file in enumerate(files):
        fs,aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0,len(aud)-audlen,audstr):
            wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])

        print(idx,file)

## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":

    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    # f = open('lists/fileparts.txt','r')
    # fileparts = f.readlines()
    # f.close()
    fileparts = [ # these are from https://github.com/clovaai/voxceleb_trainer/tree/master/lists
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partaa e395d020928bc15670b570a21695ed96'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partab bbfaaccefab65d82b21903e81a8a8020'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partac 017d579a2a96a077f40042ec33e51512'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_dev_wav_partad 7bb1e9f70fddc7a678fa998ea8b3ba19'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partaa da070494c573e5c0564b1d11c3b20577'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partab 17fe6dab2b32b48abaf1676429cdd06f'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partac 1de58e086c5edf63625af1cb6d831528'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partad 5a043eb03e15c5a918ee6a52aad477f9'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partae cea401b624983e2d0b2a87fb5d59aa60'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partaf fc886d9ba90ab88e7880ee98effd6ae9'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partag d160ecc3f6ee3eed54d55349531cb42e'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_dev_aac_partah 6b84a81b9af72a9d9eecbb3b1f602e65'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox1_test_wav.zip 185fdc63c3c739954633d50379a3d102'
        'http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_test_aac.zip 0d2b3ea430a821c33263b5ea37ede312'
    ]

    # f = open('lists/files.txt','r')
    # files = f.readlines()
    # f.close()
    files = [ # these are from https://github.com/clovaai/voxceleb_trainer/tree/master/lists
        'vox1_dev_wav_parta* vox1_dev_wav.zip ae63e55b951748cc486645f532ba230b'
        'vox2_dev_aac_parta* vox2_dev_aac.zip bbc063c46078a602ca71605645c2a402'
    ]

    # f = open('lists/augment.txt','r')
    # augfiles = f.readlines()
    # f.close()
    augfiles = [ # these are from https://github.com/clovaai/voxceleb_trainer/tree/master/lists
        'http://www.openslr.org/resources/28/rirs_noises.zip e6f48e257286e05de56413b4779d8ffb',
        'http://www.openslr.org/resources/17/musan.tar.gz 0c472d4fc0c5141eca47ad1ffeb2a7df'
        ]

    if args.augment:
        download(args,augfiles)
        part_extract(args,os.path.join(args.save_path,'rirs_noises.zip'),['RIRS_NOISES/simulated_rirs/mediumroom','RIRS_NOISES/simulated_rirs/smallroom'])
        full_extract(args,os.path.join(args.save_path,'musan.tar.gz'))
        split_musan(args)

    if args.download:
        download(args,fileparts)

    if args.extract:
        # ## ========== ===========
        # ## Original script 
        # ## >> output_dir: [wav/, aac/]
        # ## ========== ===========
        # concatenate(args, files)
        # for file in files:
        #     full_extract(args,os.path.join(args.save_path,file.split()[1]))
        # out = subprocess.call('mv %s/dev/aac/* %s/aac/ && rm -r %s/dev' %(args.save_path, args.save_path, args.save_path), shell=True)
        # out = subprocess.call('mv %s/wav %s/voxceleb1' %(args.save_path,args.save_path), shell=True)
        # out = subprocess.call('mv %s/aac %s/voxceleb2' %(args.save_path,args.save_path), shell=True)

        ## ========== ===========
        ## Custom script1 -- if you just downloaded ...parta, b, c, ... right ahead
        ## >> output_dir: [vox1_dev/, vox1_test/, vox2_dev/, vox2_test/]
        ## ========== ===========
        concatenate(args, files)
        for file in files:
            fname = file.split()[1]
            output_dir = fname.replace('.zip', '')
            full_extract(args,os.path.join(args.save_path, fname))
        
            if 'vox1' in fname: # >> vox1_test_wav/, vox1_dev_wav/
                out = subprocess.call('mv %s/wav %s/%s' % (args.save_path, args.save_path, output_dir), shell=True)
            elif 'vox2_test_aac' in fname: # >> vox2_test_aac/
                out = subprocess.call('mv %s/aac %s/%s' % (args.save_path, args.save_path, output_dir), shell=True)
            elif 'vox2_dev_aac' in fname: # >> vox2_test_aac/
                out = subprocess.call('mv %s/dev/aac %s/%s && rm -r %s/dev' %(args.save_path, args.save_path, output_dir, args.save_path), shell=True)

        # ## ========== ===========
        # ## Custom script2 -- for extracting the final (.zip) files only
        # # ## >> output_dir: [vox1_dev/, vox1_test/, vox2_dev/, vox2_test/]        
        # ## ========== ===========
        # fnames = [i for i in os.listdir(args.save_path) if i.endswith('.zip')]
        # for fname in fnames:
        #     output_dir = fname.replace('.zip', '')
        #     full_extract(args, os.path.join(args.save_path, fname))
            
        #     if 'vox1' in fname: # >> vox1_test_wav/, vox1_dev_wav/
        #         out = subprocess.call('mv %s/wav %s/%s' % (args.save_path, args.save_path, output_dir), shell=True)
        #     elif 'vox2_test_aac' in fname: # >> vox2_test_aac/
        #         out = subprocess.call('mv %s/aac %s/%s' % (args.save_path, args.save_path, output_dir), shell=True)
        #     elif 'vox2_dev_aac' in fname: # >> vox2_test_aac/
        #         out = subprocess.call('mv %s/dev/aac %s/%s && rm -r %s/dev' %(args.save_path, args.save_path, output_dir, args.save_path), shell=True)


    if args.convert:
        # ## ========== ===========
        # ## Original script 
        # ## ========== ===========
        # convert(args)
    
        ## ========== ===========
        ## Custom script
        ## ========== ===========
        print('vox2_test_aac:', len(os.listdir('%s/vox2_test_aac' % args.save_path)), 'speakers,', len(glob.glob('%s/vox2_test_aac/*/*/*.m4a' % args.save_path)), 'files retrieved')
        print('vox2_dev_aac :', len(os.listdir('%s/vox2_dev_aac' % args.save_path)), 'speakers,', len(glob.glob('%s/vox2_dev_aac/*/*/*.m4a' % args.save_path)), 'files retrieved')

        fpaths = glob.glob('%s/vox2_*_aac/*/*/*.m4a' % args.save_path)
        fpaths.sort()

        print('Converting vox2_test/dev files from AAC to WAV')        
        if args.ncpu > 1:
            freeze_support()
            with Pool(args.ncpu) as pool:
                list(tqdm(pool.imap(convert, fpaths), total=len(fpaths)))
                
        else:
            for fpath in tqdm(fpaths):
                convert(args)
                
        out = subprocess.call('mv %s/vox2_test_aac %s/vox2_test_wav' % (args.save_path, args.save_path), shell=True)
        out = subprocess.call('mv %s/vox2_dev_aac %s/vox2_dev_wav' % (args.save_path, args.save_path), shell=True)
        

    ## ========== ===========
    ## Custom script
    ## ========== ===========
    if args.preprocess:
        outdir_names = ['vox1_dev_wav', 'vox1_test_wav', 'vox2_dev_wav', 'vox2_test_wav']
        
        print('Collecting speaker list from each splits.')
        for dirname in outdir_names:
            speaker_list = os.listdir('%s/%s' % (args.save_path, dirname))
                        
            with open('speakers/%s' % (dirname.replace('_wav', '.txt')), 'w') as f:
                for s_id in speaker_list:
                    f.write(s_id + '\n')
        
        os.makedirs('%s/voxceleb1' % args.save_path, exist_ok=True)
        # out = subprocess.call('mv %s/vox1_test_wav/* %s/voxceleb1/' % (args.save_path, args.save_path), shell=True)
        out = subprocess.call('mv %s/vox1_dev_wav/* %s/voxceleb1/' % (args.save_path, args.save_path), shell=True)

        os.makedirs('%s/voxceleb2' % args.save_path, exist_ok=True)
        out = subprocess.call('mv %s/vox2_test_wav/* %s/voxceleb2/' % (args.save_path, args.save_path), shell=True)
        out = subprocess.call('mv %s/vox2_dev_wav/* %s/voxceleb2/' % (args.save_path, args.save_path), shell=True)
                    
                
