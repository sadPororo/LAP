import time
import random
import neptune
import numpy as np

import torch
import torch.distributed as dist

from utils.loggers import printlog
from utils.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from typing import Union
from tqdm import tqdm

from evaluate.common import *
from train.common import to_device, log_model_metainfo, log_results
from sklearn.linear_model import LogisticRegression
from evaluate.norm_score import normalized_score




def qmf_input(embed_e:torch.Tensor, embed_t:torch.Tensor, duration_e:float, duration_t:float, cohort_speaker_embeddings:torch.Tensor, cohort_mean_vec:torch.Tensor, cohort_size:int) -> np.array:
    """ get qmf input """
    
    # similarity score (apply as-norm)
    submean_embed_e = (embed_e - cohort_mean_vec).unsqueeze(0) # (1, embed_size)
    submean_embed_t = (embed_t - cohort_mean_vec).unsqueeze(0)
    norm_score = normalized_score(submean_embed_e, submean_embed_t, cohort_speaker_embeddings, cohort_size, cosine_similarity)
    
    # speech durations
    log_duration_e = np.log(duration_e)
    log_duration_t = np.log(duration_t)
    
    # embedding norms
    l1norm_e = np.linalg.norm(embed_e, ord=1) # (embed_size,) -> float
    l2norm_e = np.linalg.norm(embed_e, ord=2)
    l1norm_t = np.linalg.norm(embed_t, ord=1)
    l2norm_t = np.linalg.norm(embed_t, ord=2)
    
    # embedding std
    std_e = np.std(embed_e.numpy())
    std_t = np.std(embed_t.numpy())
        
    return [norm_score, log_duration_e, log_duration_t, l1norm_e, l2norm_e, l1norm_t, l2norm_t, std_e, std_t]
    


def calculate_scores(config:dict, eval_corpus_embeddings:dict, cohort_speaker_embeddings:torch.Tensor, cohort_mean_vec:torch.Tensor, qmf_model:LogisticRegression):
    """ calculate similarity between pairs """
    cohort_size = config.model.cohort_size if not config.args.quick_check else 100
    cohort_speaker_embeddings = cohort_speaker_embeddings - cohort_mean_vec.unsqueeze(0) # (n_cohort_speakers, embed_size)
    
    eval_scores = {}
    for trial_mode in ['vox-O', 'vox-E', 'vox-H']:
        
        ##____ escape condition
        if config.args.quick_check and trial_mode in ['vox-E', 'vox-H']:
            continue
        
        ##____ load trials
        eval_trial = read_trial(config.model.data_path, trial_mode)
        
        ##____ trial iteration
        model_input, labels = [], []
        for line in tqdm(eval_trial):
            
            label, ipath_e, ipath_t = line.split(' ')
            (_, duration_e, embed_e) = eval_corpus_embeddings[ipath_e] # (speaker_id, duration, embedding)
            (_, duration_t, embed_t) = eval_corpus_embeddings[ipath_t]
            
            model_input.append(qmf_input(embed_e, embed_t, duration_e, duration_t, cohort_speaker_embeddings, cohort_mean_vec, cohort_size))
            labels.append(int(label))
        
        scores = qmf_model.predict_proba(np.array(model_input))[:, 1] # (n_trials, 2) -> (n_trials,)
        labels = np.array(labels)
        
        ##____ evaluation metrics
        _, eer, _, _ = tuneThresholdfromScore(scores, labels, [1, 0.1])
        eval_scores[f"{trial_mode}/EER"] = eer

        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        eval_scores[f"{trial_mode}/DCF01"], eval_scores[f"{trial_mode}/DCF01_thresh"] = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)
        eval_scores[f"{trial_mode}/DCF05"], eval_scores[f"{trial_mode}/DCF05_thresh"] = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
    
    return eval_scores



def train_qmf_model(config:dict, cohort_speakerwise_embeddings:dict, cohort_speaker_to_index:dict, cohort_speaker_embeddings:torch.Tensor, cohort_mean_vec:torch.Tensor) -> LogisticRegression:
    """ train calibration model """
    ##____ configurations
    cohort_size = config.model.cohort_size if not config.args.quick_check else 100
    train_speaker_list = list(cohort_speakerwise_embeddings.keys())
    cohort_speaker_embeddings = cohort_speaker_embeddings - cohort_mean_vec.unsqueeze(0) # (n_cohort_speakers, embed_size)
    
    ##____ sample pairs to train logistic regression model
    train_input, train_label = [], []
    for i in tqdm(range(config.model.calibration_train_trials)):
        
        ##____ sample positive (target) pair
        if i < (config.model.calibration_train_trials * 0.5):
            # sample target speaker
            speaker_id = random.choice(train_speaker_list)
            current_label = 1

            # exclude self from cohort set
            current_cohort_index = np.ones(cohort_speaker_embeddings.size(0), dtype=bool) # (n_cohort_speakers,)
            current_cohort_index[cohort_speaker_to_index[speaker_id]] = False
            current_cohort_speaker_embeddings = cohort_speaker_embeddings[current_cohort_index] # (n_cohort_speakers-1, embed_size)
            
            # sample target pair
            [(duration_e, embed_e), (duration_t, embed_t)] = random.sample(cohort_speakerwise_embeddings[speaker_id], 2)
        
        ##____ sample negative (non-target) pair
        else:
            # sample non-target speaker pair
            speaker1_id, speaker2_id = random.sample(train_speaker_list, 2)
            current_label = 0
            
            # exclude selected pair from cohort set
            current_cohort_index = np.ones(cohort_speaker_embeddings.size(0), dtype=bool) # (n_cohort_speakers,)
            current_cohort_index[cohort_speaker_to_index[speaker1_id]] = False
            current_cohort_index[cohort_speaker_to_index[speaker2_id]] = False
            current_cohort_speaker_embeddings = cohort_speaker_embeddings[current_cohort_index] # (n_cohort_speakers-2, embed_size)
            
            # sample non-target pair
            (duration_e, embed_e) = random.choice(cohort_speakerwise_embeddings[speaker1_id])
            (duration_t, embed_t) = random.choice(cohort_speakerwise_embeddings[speaker2_id])
        
        ##____ get qmf model input
        train_input.append(qmf_input(embed_e, embed_t, duration_e, duration_t, current_cohort_speaker_embeddings, cohort_mean_vec, cohort_size))
        train_label.append(current_label)

    ##____ train calibration model
    qmf_model = LogisticRegression(random_state=config.model.seed, solver=config.model.solver)
    qmf_model.fit(np.array(train_input), np.array(train_label))
    
    return qmf_model
    


def gather_speaker_embeddings(corpus_embeddings:dict) -> Tuple[dict, dict, torch.Tensor]:
    """ 
    Input:
        corpus_embeddings (dict): {
            'ipath': tuple( 'speaker_id', duration, embedding ), ...
        }
    
    Returns:
        speakerwise_embeddings (dict): { 'speaker_id': list[ tuple(duration, embedding), ... ] }
        speaker_to_index (dict): { 'speaker_id': speaker_idx }
        speaker_embeddings (Tensor): (n_speakers, embed_size)
    """
    speakerwise_embeddings = {}
    
    ##____ gather speaker-wise embeddings
    for ipath in corpus_embeddings:
        (speaker_id, duration, vector) = corpus_embeddings[ipath]
        
        if speaker_id not in speakerwise_embeddings:
            speakerwise_embeddings[speaker_id] = []
        
        speakerwise_embeddings[speaker_id].append((duration, vector)) # [(duration, embedding), ...]

    ##____ get mean of speakerwise embeddings
    speaker_embeddings = {}
    for speaker_id in speakerwise_embeddings:
        speaker_embeddings[speaker_id] = torch.stack([info[-1] for info in speakerwise_embeddings[speaker_id]], dim=0).mean(dim=0) # (embed_size,)

    ##____ stack in speakers
    speaker_to_index   = {speaker_id: i for i, speaker_id in enumerate(list(speaker_embeddings.keys()))}
    speaker_embeddings = torch.stack(list(speaker_embeddings.values()), dim=0) # (n_speakers, embed_size)
    
    return speakerwise_embeddings, speaker_to_index, speaker_embeddings



def evaluate_model(config:dict, logger:Union[neptune.Run, None]):
    """ main function for model evaluation """
    start_t = time.time()
    logfile = config.out.log
    silence = not config.model.verbose
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ phase control _____________________________________________________________________________________________
    printlog("\n========================== PHASE 6: QUALITY_MEASUREMENT_FUNCTION ==========================", logfile)
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
        cohort_speakerwise_embeddings, cohort_speaker_to_index, cohort_speaker_embeddings = gather_speaker_embeddings(cohort_corpus_embeddings)

        printlog("\nTraining QMF model...", logfile)
        qmf_model = train_qmf_model(config, cohort_speakerwise_embeddings, cohort_speaker_to_index, cohort_speaker_embeddings, cohort_mean_vec)

        printlog("\nApplying calibration...", logfile)
        eval_scores = calculate_scores(config, eval_corpus_embeddings, cohort_speaker_embeddings, cohort_mean_vec, qmf_model)
        eval_scores['time'] = time.time() - start_t
    
        ##____ result logging
        log_results(config, logger, current_epoch=0, train_scores={}, eval_scores=eval_scores, best_scores=eval_scores)
    
    ##____ end of evaluation
    printlog("\n=================================== END OF EVALUATION ====================================", logfile)
    if rank==0 and logger is not None and config.args.neptune:
        logger.stop()


    
    
    
