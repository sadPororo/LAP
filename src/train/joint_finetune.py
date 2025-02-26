import shutil
import neptune

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from model import SVmodel
from utils.loggers import printlog
from utils.utility import hypwrite, hypload
from os.path import join as opj
from typing import Union

from train.common import *
from evaluate.common import *
from evaluate.naive_score import evaluate_model_on_train



def get_model(config:dict) -> nn.Module:
    """ initiate speaker model """
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
    
    ##____ keep the last epoch configuration from prev. frozen training
    model.aam_softmax.m = prev_config.model.margin
    model.aam_softmax.update_margins(prev_config.model.margin, prev_config.model.margin_penalty, prev_config.model.topK)
    
    ##____ unfreeze the frontend
    model.update_frontend_parameters()
        
    ##____ verbose
    best_scores = torch.load(opj('../res', config.args.evaluation_id, f'{best_epoch}epoch', 'eval_scores.dict'), map_location='cpu')
    printlog(f"\t> using previous exp: '{config.args.evaluation_id}'", logfile)
    printlog("\t  : {:d} epoch result, (top {:d})".format(best_epoch, choose_topN), logfile)
    
    for trial_mode in ['vox-O', 'vox-E', 'vox-H']:
        if f'{trial_mode}/EER' in best_scores:
            printlog("\t  : {:s} EER: {:.02f}%, DCF01: {:.04f}, DCF05: {:.04f}".format(
                trial_mode, best_scores[f'{trial_mode}/EER'], best_scores[f'{trial_mode}/DCF01'], best_scores[f'{trial_mode}/DCF05']), logfile)
    
    return model



def get_optimizer(config:dict, model:Union[SVmodel, DDP], silence:bool=False) -> optim.Optimizer:
    """ initiate model optimizer/scheduler """
    logfile = config.out.log
    model_context = model.module if dist.is_initialized() else model
    
    ##____ optimimzer
    optimizer = optim.Adam(model_context.aam_softmax.parameters(), lr=config.model.lr, weight_decay=config.model.weight_decay)
    optimizer.add_param_group({'params': model_context.backend.parameters()})
    optimizer.add_param_group({'params': [p for p in model_context.frontend.parameters() if p.requires_grad]}) # update frontend parameters
    
    ##____ verbose
    printlog(f"\t> Adam", logfile, silence)
    printlog("\t  : aamhead  : (lr={:0.0e}, weight_decay={:0.0e})".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['weight_decay']), logfile, silence)
    printlog("\t  : backend  : (lr={:0.0e}, weight_decay={:0.0e})".format(optimizer.param_groups[1]['lr'], optimizer.param_groups[1]['weight_decay']), logfile, silence)
    printlog("\t  : frontend : (lr={:0.0e}, weight_decay={:0.0e})".format(optimizer.param_groups[2]['lr'], optimizer.param_groups[2]['weight_decay']), logfile, silence)
    
    return optimizer



def model_trainsetup(model:Union[SVmodel, DDP]) -> None:
    model.train()
    model_context = model.module if dist.is_initialized() else model
    model_context.update_frontend_parameters() # update frontend parameters
    model.zero_grad()
    
    return
    


def train_model(config:dict, logger:Union[neptune.Run, None]):
    """ main function for training with frozen frontend model (phase 1) """
    logfile = config.out.log
    silence = not config.model.verbose
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ phase control _____________________________________________________________________________________________
    printlog("\n================================= PHASE 2: TRAIN_FINETUNE =================================", logfile)
    ##________________________________________________________________________________________________________________
    
    printlog("\nLoading datasets...", logfile)
    train_dataset  = get_train_dataset(config, silence)
    eval_dataset   = get_evaluation_dataset(config, silence)
    # cohort_dataset = get_cohort_dataset(config, silence, is_train=True)
    
    printlog("\nConfigurating loaders...", logfile)
    train_loader  = get_train_dataloader(config, train_dataset, silence)
    eval_loader   = get_evaluation_loader(config, eval_dataset, silence)
    # cohort_loader = get_cohort_loader(config, cohort_dataset, silence)
    
    printlog('\nInitializing speaker model...', logfile)
    model = get_model(config)
    swa_model, swa_flag = None, False
    log_model_metainfo(config, logger, model, silence)
    model = to_device(config, model, silence)
    
    printlog('\nSetting model optimizer/schedulers...', logfile)
    optimizer = get_optimizer(config, model, silence)
    scheduler = get_scheduler(config, optimizer, train_loader, silence)
    swa_scheduler = get_swa_scheduler(config, optimizer, train_loader, silence)
    gradscaler = GradScaler()
    
    ##____ epoch training iteration
    start_epoch = 1
    final_epoch = 3 if config.args.quick_check else config.model.epochs
    best_scores, best_tracking = {}, []
    model_context = model.module if dist.is_initialized() else model

    for current_epoch in range(start_epoch, final_epoch+1):

        ##____ verbose / phase control _______________________________________________________________________________________________________________
        printlog("\nEpoch {:03d}/{:03d} training...".format(current_epoch, final_epoch), logfile)
        printlog("\t> lrs applied: ({:0.0e}, {:0.0e}, {:0.0e}) | decays applied: ({:0.0e}, {:0.0e}, {:0.0e})".format(
            optimizer.param_groups[2]['lr'],           optimizer.param_groups[1]['lr'],           optimizer.param_groups[0]['lr'],  
            optimizer.param_groups[2]['weight_decay'], optimizer.param_groups[1]['weight_decay'], optimizer.param_groups[0]['weight_decay']), logfile)
        ##____________________________________________________________________________________________________________________________________________
        
        ##____ train setups
        train_loader.dataset.generate_iteration(config.model.seed + current_epoch)
        model_trainsetup(model)

        ##____ train 1 epoch
        if dist.is_initialized(): dist.barrier()
        train_scores, running_mean_vec = train_one_epoch(config, model, optimizer, swa_scheduler if swa_flag else scheduler, gradscaler, train_loader)
        model, model_state = sync_model_weights(model)    

        ##____ model averaging on last 10 epochs
        if config.model.apply_swa and current_epoch >= (final_epoch - config.model.swa_anneal_epochs):
            swa_flag = True
            
            if swa_model is None:
                model_context.detach()
                ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                    (1. - config.model.swa_decay) * averaged_model_parameter + config.model.swa_decay * model_parameter
                swa_model = AveragedModel(model_context, avg_fn=ema_avg_fn)
                printlog("\t> SWA model initialized.", logfile)
            else:
                swa_model.update_parameters(model_context)
            
            if swa_model.n_averaged.item() > 0:
                printlog("\t> updating SWA model bn-stats...", logfile)
                swa_model, running_mean_vec = update_bn_ddp(config, train_loader, swa_model)
                
            model, _ = sync_swa_model_weights(model, swa_model)

        ##____ evaluate model
        if dist.is_initialized(): dist.barrier()
        eval_scores, eval_corpus_embeddings, _, _ = evaluate_model_on_train(config, model, eval_loader, None, running_mean_vec)
        
        ##____ master process
        if rank==0:
            ##____ save current checkpoint
            save_checkpoint(config, current_epoch, eval_scores, optimizer.state_dict(), scheduler.state_dict(), model_context.state_dict(), eval_corpus_embeddings, running_mean_vec)
            
            ##____ update best_scores
            best_scores = update_bestshot(config, best_scores, eval_scores, best_tracking, current_epoch)

            ##____ result logging
            log_results(config, logger, current_epoch, train_scores, eval_scores, best_scores)
        
        # ##____ get back on model state for training 
        # if swa_flag: model_context.load_state_dict(model_state, strict=True)
        # if dist.is_initialized(): dist.barrier()
    
    ##____ end of iteration
    printlog("\n====================================== TRAINING END ======================================", logfile)
    if rank==0 and logger is not None and config.args.neptune:
        logger.stop()
