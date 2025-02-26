# Rethinking Leveraging Pre-Trained Multi-Layer Representations for Speaker Verification (2025)

This is a Python implementation of our paper.  

## Environment supports & Python requirements
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04+-E95420?style=for-the-badge&logo=ubuntu&logoColor=E95420)
![Python](https://img.shields.io/badge/Python-3.8.8-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=%23EE4C2C)   
* We recommend you to visit [Previous Versions (v1.12.0)](https://pytorch.org/get-started/previous-versions/#v1120) for **PyTorch** installation including torchaudio==0.12.0.

Use the [requirements.txt](/requirements.txt) to install the rest of the Python dependencies.   
**Ubuntu-Soundfile** and **conda-ffmpeg** packages would be required for downloading and preprocessing datasets, and you can install them as:

```bash
$ pip install -r requirements.txt
$ apt-get install python3-soundfile
$ conda install -c conda-forge ffmpeg
```

## Dataset Preparation
Follow ```dataprep.sh``` files from [/data/VoxCeleb](/data/VoxCeleb) ; [MUSAN](/data/MUSAN) ; [RIRs](/data/RIRs) to download and preprocess the datasets.    

* **VoxCeleb 1 & 2** [1,2]  
* **MUSAN** [3]  
* **Room Impulse Response and Noise Database (RIRs)** [4]  
> [1] A. Nagrani, et al., “VoxCeleb: A large scale speaker identification dataset,” in _Proc. Interspeech_, 2017.  
> [2] J. S. Chung, et al., “VoxCeleb2: Deep speaker recognition,” in _Proc. Interspeech_, 2018.    
> [3] D. Snyder, et al., “MUSAN: A Music, Speech, and Noise Corpus,” arXiv, 2015.  
> [4] T. Ko, et al., “A study on data augmentation of reverberant speech for robust speech recognition,” in _Proc. ICASSP_, 2017.

## Run Experiments
Log files, model weights, and configurations will be saved under [/res](/res) directory.
* The output folder will be created as ```local-YYYYMMDD-HHmmss``` format by default.
* To use **neptune.ai** logging, set your neptune configuration at [/src/config/neptune.yaml](/src/config/neptune.yaml) and add ```--neptune``` in the command line.  
  The experiment ID created at your **neptune.ai [project]** will be the name of the output directory.

**General usage examples**
```bash
$ cd src
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

## Citation
> J. S. Kim, et al., “Rethinking Leveraging Pre-Trained Multi-Layer Representations for Speaker Verification,” preprint, 2025.  
```bash
The work is currently being reviewed.
```

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.

We referred to [https://github.com/clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) repository for the codes to prepare the datasets.  
class [AAMsoftmax_IntertopK_Subcenter](/src/loss.py) is adopted from [https://github.com/wenet-e2e/wespeaker/models/projections](https://github.com/wenet-e2e/wespeaker/blob/c9ec537b53fe1e04525be74b2550ee95bed3a891/wespeaker/models/projections.py#L243) with some modifications.
