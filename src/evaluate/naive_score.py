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
from torch.utils.data import DataLoader
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
from typing import Union
from tqdm import tqdm

from itertools import cycle, combinations, product

import neptune

from evaluate.common import *
from train.common import to_device, log_model_metainfo, log_results



def calculate_scores(config:dict, eval_corpus_embeddings:dict, cohort_mean_vec:torch.Tensor):
    """ caluclate similarity between pairs """
    
    eval_scores = {}
    for trial_mode in ['vox-O', 'vox-E', 'vox-H']:
        
        ##____ escape condition
        if config.args.quick_check and trial_mode in ['vox-E', 'vox-H']:
            continue
        
        ##____ load trials
        eval_trial = read_trial(config.model.data_path, trial_mode)
        
        scores = []
        labels = []
        
        ##____ trial iteration
        for line in tqdm(eval_trial):
            
            label, ipath_e, ipath_t = line.split(' ')
            embed_e = eval_corpus_embeddings[ipath_e][-1] # (embed_size,)
            embed_t = eval_corpus_embeddings[ipath_t][-1]
            
            ##____ sub mean on cohort set
            embed_e = (embed_e - cohort_mean_vec).unsqueeze(0) # (1, embed_size)
            embed_t = (embed_t - cohort_mean_vec).unsqueeze(0)
            
            ##____ calculate scores
            scores.append(cosine_similarity(embed_e, embed_t).item())            
            labels.append(int(label))
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        ##____ rescale scores to range in [0-1]
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        ##____ evaluation metrics
        _, eer, _, _ = tuneThresholdfromScore(scores, labels, [1, 0.1])
        eval_scores[f"{trial_mode}/EER"] = eer

        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        eval_scores[f"{trial_mode}/DCF01"], eval_scores[f"{trial_mode}/DCF01_thresh"] = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)
        eval_scores[f"{trial_mode}/DCF05"], eval_scores[f"{trial_mode}/DCF05_thresh"] = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
    
    return eval_scores



def evaluate_model_on_train(config:dict, model:Union[SVmodel, nn.Module, DDP], eval_loader:DataLoader, cohort_loader:Union[DataLoader, None], running_mean_vec:torch.Tensor=None) -> Tuple[dict, dict, Union[dict, None], torch.Tensor]:
    """ basic model evaluation 
    
    Returns:
        eval_scores (dict): {
            'time': total time-consumed for evaluation
            'vox-O/EER': ...,
            ...
        }
    """
    start_t = time.time()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    eval_corpus_embeddings, _ = extract_embeddings(config, model, eval_loader)
    
    if running_mean_vec is not None: # skip cohort embedding extraction if running mean is given while training
        cohort_corpus_embeddings = None
        cohort_mean_vec = running_mean_vec
    else:
        cohort_corpus_embeddings, cohort_mean_vec = extract_embeddings(config, model, cohort_loader)
    
    eval_scores = {}
    if rank==0:
        eval_scores = calculate_scores(config, eval_corpus_embeddings, cohort_mean_vec)
    
    if dist.is_initialized(): dist.barrier()
    torch.cuda.empty_cache()
    eval_scores['time'] = time.time() - start_t

    return eval_scores, eval_corpus_embeddings, cohort_corpus_embeddings, cohort_mean_vec



def evaluate_model(config:dict, logger:Union[neptune.Run, None]):
    """ main function for naive model evaluation """
    start_t = time.time()
    logfile = config.out.log
    silence = not config.model.verbose
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ phase control _____________________________________________________________________________________________
    printlog("\n============================= PHASE 4: NAIVE_SCORE_EVALUATION =============================", logfile)
    ##________________________________________________________________________________________________________________

    printlog('\nInitializing speaker model...', logfile)
    model = get_eval_model(config)
    log_model_metainfo(config, logger, model, silence=True)
    model = to_device(config, model, silence)    
    
    printlog("\nLoading datasets...", logfile)
    ##____ load embeddings if exists
    eval_corpus_embeddings = get_eval_corpus_embeddings(config, model)
    cohort_corpus_embeddings, cohort_mean_vec = get_cohort_corpus_embeddings(config, model)
    
    ##____ score metric
    eval_scores = {}
    if rank==0:
        eval_scores = calculate_scores(config, eval_corpus_embeddings, cohort_mean_vec)        
        eval_scores['time'] = time.time() - start_t
    
        ##____ result logging
        log_results(config, logger, current_epoch=0, train_scores={}, eval_scores=eval_scores, best_scores=eval_scores)
    
    ##____ end of evaluation
    printlog("\n=================================== END OF EVALUATION ====================================", logfile)
    if rank==0 and logger is not None and config.args.neptune:
        logger.stop()


