
#%%
import os
import copy
import time
import random
import shutil
# import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

import soundfile as sf
import numpy as np

from model import SVmodel
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import autocast

# from multiprocessing import Pool, freeze_support
from sklearn.preprocessing import StandardScaler
from utils.utility import hypwrite, hypload
from utils.loggers import printlog
from utils.dataset import EmbeddingExtractionDataset, load_voxceleb_corpus
from utils.sampler import DistributedEvalSampler
from utils.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from os.path import join as opj
from typing import Union, Tuple
from tqdm import tqdm

from itertools import cycle, combinations, product
from collections import OrderedDict

import neptune



def cosine_similarity(a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    """ 
    Input:
        a (Tensor): 1 x embed_size
        b (Tensor): n x embed_size
    
    Returns:
        similarity (Tensor): n, range in [-1. - 1.]
    """
    assert a.size(0)==1
    return F.cosine_similarity(a, b)


def cosine_distance(a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    """ 
    Input:
        a (Tensor): 1 x embed_size
        b (Tensor): n x embed_size
    
    Returns:
        distance (Tensor): n
    """
    assert a.size(0)==1
    return -1. * torch.cdist(a, b).squeeze(0)



def read_trial(data_path:str, mode:str='vox-O') -> list:
    """ read trial.txt file
    
    Returns:
        trials (list): ["LABEL ENROLL_PATH TEST_PATH", ... ]
    """
    
    if mode=='vox-O':   fname='veri_test2.txt'
    elif mode=='vox-E': fname='list_test_all2.txt'
    elif mode=='vox-H': fname='list_test_hard2.txt'
    else: raise NotImplementedError(mode)
    
    ##____read trials
    with open(f"{data_path}/trials/{fname}", 'r') as f:
        trials = [line.strip() for line in f.readlines()]
        
    return trials



def get_evaluation_dataset(config:dict, silence:bool=False) -> EmbeddingExtractionDataset:
    """ init voxceleb evaluation corpus (voxceleb1) """
    logfile = config.out.log
    
    ##____ read filepaths/speakers in Vox1 & 2
    vox1_dev_corpus,  vox1_dev_speakers  = load_voxceleb_corpus(config.model.data_path, 'voxceleb1', 'dev')
    vox1_test_corpus, vox1_test_speakers = load_voxceleb_corpus(config.model.data_path, 'voxceleb1', 'test')
    # vox2_dev_corpus,  vox2_dev_speakers  = load_voxceleb_corpus(config.model.data_path, 'voxceleb2', 'dev')
    # vox2_test_corpus, vox2_test_speakers = load_voxceleb_corpus(config.model.data_path, 'voxceleb2', 'test')

    ##____ evaluation data configuration
    if config.args.quick_check:
        eval_corpus   = vox1_test_corpus  # vox-O
        eval_speakers = vox1_test_speakers
    else:
        eval_corpus   = vox1_dev_corpus   + vox1_test_corpus  # vox-O, vox-H, vox-E
        eval_speakers = vox1_dev_speakers + vox1_test_speakers
    
    eval_dataset = EmbeddingExtractionDataset(eval_corpus, eval_speakers)
    
    ##____ verbose
    printlog(f"\t> evaluation data: voxceleb1 corpus", logfile, silence)
    printlog(f"\t  : {len(eval_dataset):,} utterance pairs", logfile, silence)
    
    return eval_dataset



def get_cohort_dataset(config:dict, silence:bool=False, is_train:bool=False) -> EmbeddingExtractionDataset:
    """ init voxceleb cohort corpus (voxceleb1) """
    logfile = config.out.log
    
    ##____ read filepaths/speakers in Vox1 & 2
    vox1_dev_corpus,  vox1_dev_speakers  = load_voxceleb_corpus(config.model.data_path, 'voxceleb1', 'dev')
    # vox1_test_corpus, vox1_test_speakers = load_voxceleb_corpus(config.model.data_path, 'voxceleb1', 'test')
    vox2_dev_corpus,  vox2_dev_speakers  = load_voxceleb_corpus(config.model.data_path, 'voxceleb2', 'dev')
    vox2_test_corpus, vox2_test_speakers = load_voxceleb_corpus(config.model.data_path, 'voxceleb2', 'test')
    
    #____ evaluation data configuration
    if config.args.quick_check or config.model.cohort_set=='voxceleb2-test':
        cohort_corpus   = vox2_test_corpus  # 118 speakers
        cohort_speakers = vox2_test_speakers
    elif config.model.cohort_set=='voxceleb2-dev': # 5994 speakers
        cohort_corpus   = vox2_dev_corpus
        cohort_speakers = vox2_dev_speakers
    elif config.model.cohort_set=='voxceleb12-dev': # 7205 speakers
        cohort_corpus   = vox1_dev_corpus   + vox2_dev_corpus
        cohort_speakers = vox1_dev_speakers + vox2_dev_speakers
    else: raise NotImplementedError(config.model.cohort_set)

    ##____ configure cohort audio max length
    max_duration = (config.model.train_max_duration * 2) if is_train else config.model.cohort_max_duration
    cohort_dataset = EmbeddingExtractionDataset(cohort_corpus, cohort_speakers, config.model.sample_rate, max_duration)

    ##____ verbose
    printlog(f"\t> cohort data: {config.model.cohort_set}", logfile, silence)
    printlog(f"\t  : {cohort_dataset.n_speakers():,} speakers, {cohort_dataset.n_utterances():,} utterances, max duration {cohort_dataset.max_duration()}s", logfile, silence)
    
    return cohort_dataset



def get_evaluation_loader(config:dict, eval_dataset:Dataset, silence:bool=False) -> DataLoader:
    """ initiate dataloader """
    logfile = config.out.log

    ##____ configure sampler
    if dist.is_initialized():
            dist_sampler = DistributedEvalSampler(dataset=eval_dataset, shuffle=False) # for evaluation
    else:   dist_sampler = None        
    
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=False, sampler=dist_sampler)

    ##____ verbose
    printlog(f"\t> Evaluation set loader", logfile, silence)
    printlog(f"\t  : batch_size=1, num_workers=1", logfile, silence)
    
    return eval_loader



def get_cohort_loader(config:dict, cohort_dataset:Union[Dataset, EmbeddingExtractionDataset], silence:bool=False) -> DataLoader:
    """ initiate dataloader """
    logfile = config.out.log

    ##____ configure sampler
    if dist.is_initialized():
            dist_sampler = DistributedEvalSampler(dataset=cohort_dataset, shuffle=False) # for evaluation
    else:   dist_sampler = None

    ##____ batch size    
    if cohort_dataset.max_sec is None: # no max limit is set, varying length instance
        ngpu = len(config.gpus)
        proc_batch_size = 1
        ncpu = 1
        pin_memory = False
        
    else: # max duration is set, multiple instance can be batched
        if dist.is_initialized():
            ngpu = len(config.gpus)
            ncpu = config.model.ncpu // ngpu
            proc_batch_size = config.model.cohort_batch_size // ngpu
            pin_memory = True
        else:
            ngpu = 1
            ncpu = config.model.ncpu
            proc_batch_size = config.model.cohort_batch_size
            pin_memory = False
    
    ##____ verbose
    printlog(f"\t> Cohort set loader", logfile, silence)
    printlog(f"\t  : CPU usage : {ngpu * ncpu} cores", logfile, silence)
    printlog(f"\t  : batch size : {proc_batch_size * ngpu}", logfile, silence)
    
    cohort_loader = DataLoader(dataset=cohort_dataset, batch_size=proc_batch_size, num_workers=ncpu, shuffle=False, pin_memory=pin_memory, sampler=dist_sampler)
    
    return cohort_loader



def extract_embeddings(config:dict, model:Union[SVmodel, nn.Module, DDP], dataloader:DataLoader) -> Tuple[dict, torch.Tensor]:
    """ extract speaker embeddings from given utterances
    
    Returns:
        corpus_embeddings (dict): { "SPEAKER_ID/VIDEO_ID/UTTERANCE_IDX.wav": tuple( "SPEAKER_ID", UTTERNACE_DURATION, EMBEDDING_VECTOR ), ... }
        corpus_mean_vec (Tensor): (embed_size,)
    """
    sr = config.model.sample_rate
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    model_context = model.module if dist.is_initialized() else model

    model.eval()
    corpus_embeddings = {}
    corpus_mean_vec   = torch.zeros(model_context.embed_size)
    corpus_size       = 0
    
    ##____ evaluation
    with torch.no_grad():
        for i, (waveform, length, ipath, label) in enumerate(tqdm(dataloader)):
            """ 
            waveform   (FloatTensor): (B, max_length)
            length     (LongTensor): (B,)
            ipath      (list): [ "SPEAKER_ID/VIDEO_ID/UTTERANCE_NO.wav", ...]
            label      (LongTensor): (B,)
            """
            ##____ model forward
            embeddings = model(waveform.to(rank)).cpu() # (B, embed_size)
            
            ##____ gather utterance-wise embeddings
            for j, vector in enumerate(embeddings):
                speaker_id = ipath[j].split('/')[0]
                duration   = length[j].item() / sr
                corpus_embeddings[ipath[j]] = (speaker_id, duration, vector)
                
                corpus_mean_vec += vector
                corpus_size += 1
            
    ##____ end of iteration, if DDP: gather all embeddings from subproc's
    if dist.is_initialized():
        torch.save((corpus_embeddings, corpus_mean_vec, corpus_size), opj(config.out.dir, f"corpus_embeddings_part.rank{rank}_gpu{config.gpus[rank]}"))
        dist.barrier()
        
        if rank==0:
            ##____ master process gather
            corpus_embeddings = {}
            corpus_mean_vec   = torch.zeros(model_context.embed_size)
            corpus_size       = 0
            
            for rank_id, gpu_id in enumerate(config.gpus):
                (part_embeddings, part_mean_vec, part_size) = torch.load(opj(config.out.dir, f"corpus_embeddings_part.rank{rank_id}_gpu{gpu_id}"))

                for ipath in part_embeddings:
                    corpus_embeddings[ipath] = part_embeddings[ipath]
                corpus_mean_vec += part_mean_vec
                corpus_size     += part_size

                os.remove(opj(config.out.dir, f"corpus_embeddings_part.rank{rank_id}_gpu{gpu_id}"))
            
            ##____ broadcast
            torch.save((corpus_embeddings, corpus_mean_vec, corpus_size), opj(config.out.dir, f"corpus_embeddings.tmp"))
            
        ##____ read integrated embeddings
        dist.barrier()
        (corpus_embeddings, corpus_mean_vec, corpus_size) = torch.load(opj(config.out.dir, f"corpus_embeddings.tmp"))
        
        dist.barrier()
        if rank==0: os.remove(opj(config.out.dir, f"corpus_embeddings.tmp"))
    
    ##____ get corpus mean vector
    corpus_mean_vec = corpus_mean_vec / corpus_size

    if dist.is_initialized(): dist.barrier()
    torch.cuda.empty_cache()
        
    return corpus_embeddings, corpus_mean_vec



def get_eval_model(config:dict) -> nn.Module:
    """ initiate speaker model by averaging weights from mutiple training epochs """
    choose_topN = config.model.choose_topN
    logfile = config.out.log    
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ init model from previous configuration
    prev_config = hypload(opj('../res', config.args.evaluation_id, 'model_config.yaml'))
    model = SVmodel(prev_config)
    
    if rank==0: ##____ save the previous configuration as "model_config" for the next phase
        hypwrite(prev_config, opj(config.out.dir, 'model_config.yaml'))

    ##____ read the best tracking meta
    with open(opj('../res', config.args.evaluation_id, 'best_tracking.txt'), 'r') as f:
        best_tracking = f.readlines()
        best_tracking = [line.strip().split(', ') for line in best_tracking] # [(epoch, eer), ...]
    
    ##____ load weights from previous experiments
    best_epoch = int(best_tracking[choose_topN-1][0])
    model.load_state_dict(torch.load(opj('../res', config.args.evaluation_id, f'{best_epoch}epoch', 'model.state'), map_location='cpu'), strict=True)
        
    ##____ verbose
    best_scores = torch.load(opj('../res', config.args.evaluation_id, f'{best_epoch}epoch', 'eval_scores.dict'), map_location='cpu')
    printlog(f"\t> using previous exp: '{config.args.evaluation_id}'", logfile)
    printlog("\t  : {:d} epoch result, (top {:d})".format(best_epoch, choose_topN), logfile)
    
    for trial_mode in ['vox-O', 'vox-E', 'vox-H']:
        if f'{trial_mode}/EER' in best_scores:
            printlog("\t  : {:s} EER: {:.02f}%, DCF01: {:.04f}, DCF05: {:.04f}".format(
                trial_mode, best_scores[f'{trial_mode}/EER'], best_scores[f'{trial_mode}/DCF01'], best_scores[f'{trial_mode}/DCF05']), logfile)
    
    return model



def get_eval_corpus_embeddings(config:dict, model:Union[nn.Module, DDP]) -> dict:
    """ load eval_corpus_embeddings if exist, else extract embeddings """
    logfile = config.out.log
    silence = not config.model.verbose
    n_best_models = config.model.n_best_models
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    
    ##____ load embeddings if exists
    if os.path.isfile(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'eval_corpus.embeddings')):
        printlog(f"\t> '{config.args.evaluation_id}' eval_corpus.embedding found, loading embeddings.", logfile)
        eval_corpus_embeddings = torch.load(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'eval_corpus.embeddings'))
    
    else:
        eval_dataset = get_evaluation_dataset(config, silence)
        eval_loader  = get_evaluation_loader(config, eval_dataset, silence)
        
        printlog("\t> extracting embeddings from evaluation corpus.", logfile)
        if dist.is_initialized(): dist.barrier()
        eval_corpus_embeddings, _ = extract_embeddings(config, model, eval_loader)
        
        ##____ save embeddings
        if rank==0 and not config.args.quick_check: 
            os.makedirs(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model'), exist_ok=True)
            torch.save(eval_corpus_embeddings, opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'eval_corpus.embeddings'))
    
    return eval_corpus_embeddings



def get_cohort_corpus_embeddings(config:dict, model:Union[nn.Module, DDP]) -> dict:
    """ load eval_corpus_embeddings if exist, else extract embeddings """
    logfile = config.out.log
    silence = not config.model.verbose
    n_best_models = config.model.n_best_models    
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ load embeddings if exists
    if os.path.isfile(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'cohort_corpus.embeddings')):
        printlog(f"\t> '{config.args.evaluation_id}' cohort_corpus.embedding found, loading embeddings.", logfile)
        cohort_corpus_embeddings = torch.load(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'cohort_corpus.embeddings'))
        cohort_mean_vec          = torch.load(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'cohort_mean_vec.tensor'))
    
    else:
        cohort_dataset = get_cohort_dataset(config, silence)
        cohort_loader  = get_cohort_loader(config, cohort_dataset, silence)

        printlog("\t> extracting embeddings from cohort corpus.", logfile)
        if dist.is_initialized(): dist.barrier()
        cohort_corpus_embeddings, cohort_mean_vec = extract_embeddings(config, model, cohort_loader)
        
        ##____ save embeddings
        if rank==0 and not config.args.quick_check: 
            os.makedirs(opj('../res', config.args.evaluation_id, f'best{n_best_models}_model'), exist_ok=True)            
            torch.save(cohort_corpus_embeddings, opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'cohort_corpus.embeddings'))
            torch.save(cohort_mean_vec,          opj('../res', config.args.evaluation_id, f'best{n_best_models}_model', 'cohort_mean_vec.tensor'))
    
    return cohort_corpus_embeddings, cohort_mean_vec

# %%
