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
> [1]&nbsp;&nbsp; A. Nagrani, et al., “VoxCeleb: A large scale speaker identification dataset,” in _Proc. Interspeech_, 2017.  
> [2]&nbsp;&nbsp; J. S. Chung, et al., “VoxCeleb2: Deep speaker recognition,” in _Proc. Interspeech_, 2018.    
> [3]&nbsp;&nbsp; D. Snyder, et al., “MUSAN: A Music, Speech, and Noise Corpus,” arXiv, 2015.  
> [4]&nbsp;&nbsp; T. Ko, et al., “A study on data augmentation of reverberant speech for robust speech recognition,” in _Proc. ICASSP_, 2017.

## Run Experiments
Log files, model weights, and configurations will be saved under [/res](/res) directory.
* The output folder will be created as ```local-YYYYMMDD-HHmmss``` format by default.
* To use **neptune.ai** logging, set your neptune configuration at [/src/config/neptune.yaml](/src/config/neptune.yaml) and add ```--neptune``` in the command line.  
  The experiment ID created at your **neptune.ai [project]** will be the name of the output directory.

This framework supports six-phase model training/evaluation processes.  
If you are starting the process from **phase (2-6)**, you must pass the ```--evaluation``` argument to load model weights.  

  1. **Pre-training speaker network (backend) from scratch**  
  This cold-start stage is hooked by ```--train_frozen``` argument given at the command line.  
  The optimizer only updates the backend while the frontend remains frozen.  
```bash
# Example
~/src$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train_frozen
```
  2. **Joint fine-tuning of frontend and backend networks**  
     The second stage is hooked by ```--train_finetune``` argument given at the command line.  
```bash
# Example
~/src$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train_finetune --evaluation_id 'EXP_ID'
```
  3. **Large-margin fine-tuning**  
     The stage is hooked by ```--train_finetune``` argument  
```bash
# Example
~/src$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train_lmft --evaluation_id 'EXP_ID'
```
  4. **Naive evaluation** (```--naive_evaluation```)  
     Supports cosine-similarity measurement with substitution of training speaker-embedding mean vector.  
  5. **Adaptive score normalization** (```--score_normalize```)  
     Produces normalized score of the verification trial given cohort speakers  
  6. **Quality-aware score calibration** (```--score_calibrate```)  
     We implement a linear QMF model, which considers speech durations, embedding norms, and variance of embeddings.
```bash
# Example of evaluation phases applied in one go.
~/src$ CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --naive_evaluation --score_normalize --score_calibrate --evaluation_id 'EXP_ID'
```

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

Thanks to:
* [https://github.com/clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer)  
  referred to the data preparation codes and adopted the code implementation of evaluation metrics ([/src/utils/metrics.py](/src/utils/metrics.py)).
  
* [https://github.com/wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker/blob/c9ec537b53fe1e04525be74b2550ee95bed3a891/wespeaker/models/projections.py#L243)  
  adopted the implementation for the training loss **class AAMsoftmax_IntertopK_Subcenter** ([/src/loss.py](/src/loss.py)) with slight modifications.

* [https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py)  
  adopted for the learning-rate scheduler **class CosineAnnealingWarmupRestarts** ([/src/utils/scheduler.py](/src/utils/scheduler.py)).

* [https://github.com/espnet](https://github.com/espnet/espnet/blob/master/espnet2/layers/augmentation.py#L294)  
  for the implementation of the **speaker augmentation** ([/src/utils/dataset.py](/src/utils/dataset.py#L146))

* [https://github.com/SeungjunNah/DeepDeblur-PyTorch](https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py)  
  of the customized distributed sampler at the evaluation process **class DistributedEvalSampler** ([/src/utils/sampler.py](/src/utils/sampler.py))

* [https://github.com/lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py)  
  for the implementation of the speaker model **class ECAPA_TDNN** ([/src/modules/speaker_networks/ecapa_tdnn.py](/src/modules/speaker_networks/ecapa_tdnn.py))

* [https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling](https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling)  
  for the speaker model **class MHFA** ([/src/modules/speaker_networks/mhfa.py](/src/modules/speaker_networks/mhfa.py)) with some modifications to fit this framework.


  
