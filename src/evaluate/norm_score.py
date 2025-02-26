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




def normalized_score(embed_e:torch.Tensor, embed_t:torch.Tensor, cohort_speaker_embeddings:torch.Tensor, cohort_size:int, score_func) -> float:
    """ get normalized score
    
    Input:
        embed_e (Tensor): Size(1, embed_size)
        embed_t (Tensor): Size(1, embed_size)
        cohort_speaker_embeddings (Tensor): Size(n_speakers, embed_size)
        cohort_size (int)
        score_func: cosine_similarity | cosine_distance (score metric function)
    """
    
    ##____ enroll vs. test
    e_vs_t = score_func(embed_e, embed_t) # (1,)
    
    ##____ each [enroll | test] vs. cohort set, and the k-most similar scores
    e_vs_cohort = score_func(embed_e, cohort_speaker_embeddings).topk(k=cohort_size)[0] # (topk,)
    t_vs_cohort = score_func(embed_t, cohort_speaker_embeddings).topk(k=cohort_size)[0]
    
    ##____ normalized score
    score = 0.5 * (
        (e_vs_t - e_vs_cohort.mean()) / e_vs_cohort.std() + 
        (e_vs_t - t_vs_cohort.mean()) / t_vs_cohort.std()
    ).item()
    
    return score



# def normalized_score2(embed_e:torch.Tensor, embed_t:torch.Tensor, cohort_speaker_embeddings:torch.Tensor, cohort_size:int, score_func) -> float:
#     """ get normalized score
    
#     Input:
#         embed_e (Tensor): Size(1, embed_size)
#         embed_t (Tensor): Size(1, embed_size)
#         cohort_speaker_embeddings (Tensor): Size(n_speakers, embed_size)
#         cohort_size (int)
#         score_func: cosine_similarity | cosine_distance (score metric function)
#     """
#     ##____ enroll vs. test
#     e_vs_t = score_func(embed_e, embed_t) # (1,)
    
#     ##____ each [enroll | test] vs. cohort set, and the k-most similar cohort embeddings
#     _, e_vs_cohort_indice = score_func(embed_e, cohort_speaker_embeddings).topk(k=cohort_size) # (topk,)
#     _, t_vs_cohort_indice = score_func(embed_t, cohort_speaker_embeddings).topk(k=cohort_size)
    
#     ##____ cross cohort simiilartiy scores
#     e_vs_cohort = score_func(embed_e, cohort_speaker_embeddings[e_vs_cohort_indice]) # (topk,)
#     t_vs_cohort = score_func(embed_t, cohort_speaker_embeddings[t_vs_cohort_indice])
    
#     ##____ normalized score
#     score = 0.5 * (
#         (e_vs_t - e_vs_cohort.mean()) / e_vs_cohort.std() + 
#         (e_vs_t - t_vs_cohort.mean()) / t_vs_cohort.std()
#     ).item()
    
#     return score
    


def calculate_scores(config:dict, eval_corpus_embeddings:dict, cohort_speaker_embeddings:torch.Tensor, cohort_mean_vec:torch.Tensor):
    """ caluclate similarity between pairs """
    cohort_size = config.model.cohort_size if not config.args.quick_check else 100
    cohort_speaker_embeddings = cohort_speaker_embeddings - cohort_mean_vec.unsqueeze(0) # (n_cohort_speakers, embed_size)
    
    eval_scores = {}
    for trial_mode in ['vox-O', 'vox-E', 'vox-H']:
        
        ##____ escape condition
        if config.args.quick_check and trial_mode in ['vox-E', 'vox-H']:
            continue
        
        ##____ load trials
        eval_trial = read_trial(config.model.data_path, trial_mode)
        
        scores = []
        labels  = []
        
        ##____ trial iteration
        for line in tqdm(eval_trial):
            
            label, ipath_e, ipath_t = line.split(' ')
            embed_e = eval_corpus_embeddings[ipath_e][-1] # (embed_size,)
            embed_t = eval_corpus_embeddings[ipath_t][-1]
            
            ##____ sub mean on cohort set
            embed_e = (embed_e - cohort_mean_vec).unsqueeze(0) # (1, embed_size)
            embed_t = (embed_t - cohort_mean_vec).unsqueeze(0)
            
            ##____ calculate scores
            scores.append(normalized_score(embed_e, embed_t, cohort_speaker_embeddings, cohort_size, cosine_similarity))
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


def get_speaker_embeddings(corpus_embeddings:dict) -> torch.Tensor:
    """ 
    Input:
        corpus_embeddings (dict): {
            'ipath': tuple( 'speaker_id', duration, embedding ), ...
        }
    
    Returns:
        speaker_embeddings (Tensor): (n_speakers, embed_size)
    """
    speaker_embeddings = {}
    
    ##____ gather speaker-wise embeddings
    for ipath in corpus_embeddings:
        (speaker_id, duration, vector) = corpus_embeddings[ipath]
        
        if speaker_id not in speaker_embeddings:
            speaker_embeddings[speaker_id] = []
        
        speaker_embeddings[speaker_id].append(vector) # [(embed_size,), ...]
    
    ##____ get mean vectors
    for speaker_id in speaker_embeddings:
        speaker_embeddings[speaker_id] = torch.stack(speaker_embeddings[speaker_id], dim=0).mean(dim=0) # (embed_size,)
    
    ##____ stack in speakers
    speaker_embeddings = torch.stack(list(speaker_embeddings.values()), dim=0) # (n_speakers, embed_size)
    
    return speaker_embeddings



def evaluate_model(config:dict, logger:Union[neptune.Run, None]):
    """ main function for naive model evaluation """
    start_t = time.time()
    logfile = config.out.log
    silence = not config.model.verbose
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ phase control _____________________________________________________________________________________________
    printlog("\n========================== PHASE 5: ADAPTIVE_SCORE_NORMALIZATION ==========================", logfile)
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
        cohort_speaker_embeddings = get_speaker_embeddings(cohort_corpus_embeddings)
        eval_scores = calculate_scores(config, eval_corpus_embeddings, cohort_speaker_embeddings, cohort_mean_vec)        
        eval_scores['time'] = time.time() - start_t
    
        ##____ result logging
        log_results(config, logger, current_epoch=0, train_scores={}, eval_scores=eval_scores, best_scores=eval_scores)
    
    ##____ end of evaluation
    printlog("\n=================================== END OF EVALUATION ====================================", logfile)
    if rank==0 and logger is not None and config.args.neptune:
        logger.stop()


    
    
    
