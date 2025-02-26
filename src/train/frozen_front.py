import neptune
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel

from model import SVmodel
from utils.loggers import printlog
from utils.utility import hypwrite
from os.path import join as opj
from typing import Union

from train.common import *
from evaluate.common import *
from evaluate.naive_score import evaluate_model_on_train



def get_model(config:dict) -> nn.Module:
    """ initiate speaker model """
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ init model and freeze the frontend
    model = SVmodel(config)
    model.freeze_frontend_parameters()
    
    if rank==0: ##____ save the previous configuration as "model_config" for the next phase
        hypwrite(config, opj(config.out.dir, 'model_config.yaml'))
    
    return model



def get_optimizer(config:dict, model:Union[SVmodel, DDP], silence:bool=False) -> optim.Optimizer:
    """ initiate model optimizer/scheduler """
    logfile = config.out.log
    model_context = model.module if dist.is_initialized() else model
    
    ##____ optimimzer
    optimizer = optim.Adam(model_context.aam_softmax.parameters(), lr=config.model.lr, weight_decay=config.model.weight_decay)
    optimizer.add_param_group({'params': model_context.backend.parameters()})
    
    ##____ verbose
    printlog(f"\t> Adam", logfile, silence)
    printlog("\t  : aamhead  : (lr={:0.0e}, weight_decay={:0.0e})".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['weight_decay']), logfile, silence)
    printlog("\t  : backend  : (lr={:0.0e}, weight_decay={:0.0e})".format(optimizer.param_groups[1]['lr'], optimizer.param_groups[1]['weight_decay']), logfile, silence)
    
    return optimizer


def get_margin_scheduler(config:dict, train_loader:DataLoader) -> list:
    logfile = config.out.log
    silence = not config.model.verbose
    
    margin_up_steps = len(train_loader) * config.model.margin_up_epochs
    if config.model.margin_up_func == 'log':
        margins = np.log(np.linspace(5e-3, 1., num=margin_up_steps))
    
    elif config.model.margin_up_func == 'cos':
        margins = np.cos(np.linspace(-np.pi, 0., num=margin_up_steps))
    
    elif config.model.margin_up_func == 'exp':
        margins = np.exp(np.linspace(1e-1, 5, num=margin_up_steps))

    margins = (margins - margins.min()) / (margins.max() - margins.min()) * config.model.margin
    margins = np.concatenate([np.array([margins[0]] * len(train_loader) * config.model.margin_up_start), margins])
    margins = np.concatenate([margins, np.array([margins[-1]] * len(train_loader) * (config.model.epochs - config.model.margin_up_start - config.model.margin_up_epochs))])
    margins = np.array_split(margins, config.model.epochs)
    
    return margins




def model_trainsetup(config:dict, model:Union[SVmodel, DDP], current_epoch:int) -> None:
    logfile = config.out.log
    silence = not config.model.verbose

    model.train()
    model_context = model.module if dist.is_initialized() else model
    
    ##____ freeze the frontend module
    model_context.freeze_frontend_parameters()    
    model.zero_grad()
    
    return



def train_model(config:dict, logger:Union[neptune.Run, None]):
    """ main function for training with frozen frontend model (phase 1) """
    logfile = config.out.log
    silence = not config.model.verbose
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    ##____ phase control _____________________________________________________________________________________________
    printlog("\n================================== PHASE 1: TRAIN_FROZEN ==================================", logfile)
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
    margin_schedule = get_margin_scheduler(config, train_loader)
    gradscaler = GradScaler()
        
    ##____ epoch training iteration
    start_epoch = 1
    final_epoch = 3 if config.args.quick_check else config.model.epochs
    best_scores, best_tracking = {}, []
    model_context = model.module if dist.is_initialized() else model
        
    for current_epoch in range(start_epoch, final_epoch+1):

        ##____ verbose / phase control ____________________________________________________________________
        printlog("\nEpoch {:03d}/{:03d} training...".format(current_epoch, final_epoch), logfile)
        printlog("\t> lrs applied: (-, {:0.0e}, {:0.0e}) | decays applied: (-, {:0.0e}, {:0.0e})".format(
            optimizer.param_groups[1]['lr'],           optimizer.param_groups[0]['lr'],  
            optimizer.param_groups[1]['weight_decay'], optimizer.param_groups[0]['weight_decay']), logfile)
        ##_________________________________________________________________________________________________

        ##____ train setups
        train_loader.dataset.generate_iteration(config.model.seed + current_epoch)
        model_trainsetup(config, model, current_epoch)

        ##____ train 1 epoch
        if dist.is_initialized(): dist.barrier()
        train_scores, running_mean_vec = train_one_epoch(config, model, optimizer, swa_scheduler if swa_flag else scheduler, gradscaler, train_loader, margin_schedule[current_epoch-1])
        model, model_state = sync_model_weights(model)    
                
        ##____ model averaging on last '--swa_anneal_epochs' epochs
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
        if swa_flag: model_context.load_state_dict(model_state, strict=True)
        if dist.is_initialized(): dist.barrier()
        
    ##____ end of iteration
    printlog("\n====================================== TRAINING END ======================================", logfile)
        

    if rank==0 and logger is not None and config.args.neptune:
        logger.stop()
            
            
            
        


    
    
    
    
    


    