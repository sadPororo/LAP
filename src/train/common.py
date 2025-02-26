import os
import time
import numpy
import shutil
import neptune

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.optim.lr_scheduler as LrScheduler

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import SVmodel
from utils.loggers import printlog
from utils.metrics import get_Accuracy, get_F1score
from utils.dataset import TrainDataset, load_voxceleb_corpus
from utils.scheduler import CosineAnnealingWarmupRestarts

from torch.nn.utils import clip_grad_norm_
from os.path import join as opj
from typing import Union, Tuple
from tqdm import tqdm



def get_train_dataset(config:dict, silence:bool=False) -> TrainDataset:
    """ init voxceleb train dataset """
    logfile = config.out.log

    ##____ read filepaths/speakers in Vox1 & 2
    vox1_dev_corpus,  vox1_dev_speakers  = load_voxceleb_corpus(config.model.data_path, 'voxceleb1', 'dev')
    # vox1_test_corpus, vox1_test_speakers = load_voxceleb_corpus(config.model.data_path, 'voxceleb1', 'test')
    vox2_dev_corpus,  vox2_dev_speakers  = load_voxceleb_corpus(config.model.data_path, 'voxceleb2', 'dev')
    # vox2_test_corpus, vox2_test_speakers = load_voxceleb_corpus(config.model.data_path, 'voxceleb2', 'test')
    
    ##___ training set configuration
    if config.model.training_set == 'voxceleb1-dev': # 1211 speakers
        train_corpus   = vox1_dev_corpus
        train_speakers = vox1_dev_speakers
    elif config.model.training_set == 'voxceleb2-dev': # 5994 speakers
        train_corpus   = vox2_dev_corpus
        train_speakers = vox2_dev_speakers
    elif config.model.training_set == 'voxceleb12-dev': # 7205 speakers
        train_corpus   = vox1_dev_corpus   + vox2_dev_corpus
        train_speakers = vox1_dev_speakers + vox2_dev_speakers
    else: raise NotImplementedError(config.model.training_set)
    
    train_dataset = TrainDataset(config, train_corpus, train_speakers)

    ##____ verbose
    printlog(f"\t> training data: {train_dataset.training_set}", logfile, silence)
    printlog(f"\t  : {train_dataset.n_speakers():,} speakers, {train_dataset.n_utterances():,} utterances, max duration {train_dataset.max_sec}s", logfile, silence)
    if train_dataset.n_speed_perturbation() > 1:
        printlog(f"\t  : speed perturbation {train_dataset.speed_perturb} applied", logfile, silence)
    
    return train_dataset



def get_train_dataloader(config:dict, train_dataset:Dataset, silence:bool=False) -> DataLoader:
    """ initiate train dataloader """
    logfile = config.out.log

    ##____ configure sampler
    if dist.is_initialized():
            dist_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    else:   dist_sampler = None
    
    ##____ setup loader arguments
    grad_acc = config.model.grad_acc
    total_batch_size = config.model.batch_size * grad_acc
    
    if dist.is_initialized():
        ngpu = len(config.gpus)
        ncpu = config.model.ncpu // ngpu # per subproc's
        proc_batch_size = total_batch_size // ngpu
        loader_shuffle, pin_memory = False, True
        assert proc_batch_size % grad_acc == 0, "proc_batch_size: {:d}, grad_acc: {:d}".format(proc_batch_size, grad_acc)
        
    else:
        ngpu, ncpu = 1, config.model.ncpu
        proc_batch_size = total_batch_size
        loader_shuffle, pin_memory = True, False
    
    ##____ verbose
    printlog(f"\t> Train set loader", logfile, silence)
    printlog(f"\t  : using CPU {config.model.ncpu} cores", logfile, silence)
    printlog(f"\t  : total {proc_batch_size * ngpu} training batch size", logfile, silence)
    printlog(f"\t  : (batch_size x grad_acc) = {config.model.batch_size} x {grad_acc}", logfile, silence)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=proc_batch_size, num_workers=ncpu, shuffle=loader_shuffle, pin_memory=pin_memory, sampler=dist_sampler)
    
    return train_loader



def log_model_metainfo(config:dict, logger:Union[neptune.Run, None], model:SVmodel, silence=False) -> None:
    """ make loggings of model meta info """
    logfile = config.out.log

    frontend_param_size = sum(p.numel() for p in model.frontend.parameters())
    backend_param_size  = sum(p.numel() for p in model.backend.parameters())
    aamhead_param_size  = sum(p.numel() for p in model.aam_softmax.parameters())
    
    if logger is not None and config.args.neptune:
        logger['parameters/frontend']      = model.frontend.config._name_or_path.split('/')[-1]
        logger['parameters/frontend_size'] = frontend_param_size
        logger['parameters/backend_size']  = backend_param_size
        logger['parameters/clsproj_size']  = aamhead_param_size

    ##____ verbose
    printlog(f"\t> frontend module", logfile, silence)
    printlog(f"\t  : config name: '{model.frontend.config._name_or_path}'", logfile, silence)
    printlog(f"\t  : #num params: {frontend_param_size:,}", logfile, silence)
    printlog(f"\t> backend module", logfile, silence)
    printlog(f"\t  : #num params: {backend_param_size:,}", logfile, silence)
    printlog(f"\t> aam-softmax", logfile, silence)
    printlog(f"\t  : margin: {model.aam_softmax.m}, scale: {model.aam_softmax.scale}," +
             f" n_subcenter: {model.aam_softmax.n_subcenter}," +
             f" topK: {model.aam_softmax.topK}, penalty margin: {model.aam_softmax.margin_penalty}", logfile, silence)
    printlog(f"\t  : #num params: {aamhead_param_size:,}", logfile, silence)

    return



def to_device(config:dict, model:nn.Module, silence:bool=False) -> Union[nn.Module, DDP]:
    """ load model on cuda device """
    logfile = config.out.log
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    model = model.cuda(rank)
    if dist.is_initialized():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(module=model, device_ids=[rank], find_unused_parameters=True)
        ##____ verbose
        printlog(f"\t> DDP, switch nn.BatchNorm() into nn.SyncBatchNorm()", logfile, silence)
    
    return model



def get_scheduler(config:dict, optimizer:optim.Optimizer, train_loader:DataLoader, silence:bool=False) -> LrScheduler:
    """ configure scheduler """
    logfile = config.out.log
        
    ##____ scheduler
    first_cycle_steps = (config.model.epochs * len(train_loader)) // config.model.n_cycle
    warmup_steps      = int(first_cycle_steps * config.model.warmup_ratio)
    min_lr            = 0.5 * config.model.max_lr if config.args.quick_check else config.model.min_lr
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=config.model.cycle_mult, 
                                              max_lr=config.model.max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=config.model.gamma)
    
    ##____ verbose
    printlog(f"\t> CosineAnnealingWarmupRestarts", logfile, silence)
    printlog( "\t  : first_cycle_steps : {:,}".format(scheduler.first_cycle_steps), logfile, silence)
    printlog( "\t  : warmup_steps : {:}, cycle_mult : {:.02f}".format(scheduler.warmup_steps, scheduler.cycle_mult), logfile, silence)
    printlog( "\t  : max_lr : {:0.0e}, min_lr : {:0.0e}, gamma : {:.02f}".format(scheduler.max_lr, scheduler.min_lr, scheduler.gamma), logfile, silence)

    return scheduler



def get_swa_scheduler(config:dict, optimizer:optim.Optimizer, train_loader:DataLoader, silence:bool=False) -> SWALR:
    """ configure swa-scheduler """
    logfile = config.out.log

    if config.model.apply_swa:
        ##____ SWA scheduler
        anneal_steps = config.model.swa_anneal_epochs * len(train_loader)
        swa_scheduler = SWALR(optimizer, swa_lr=config.model.swa_lr, anneal_epochs=anneal_steps, anneal_strategy=config.model.swa_anneal_strategy)
                
        ##____ verbose
        printlog(f"\t> SWALR", logfile, silence)
        printlog( "\t  : swa_lr : {:}".format(swa_scheduler.get_last_lr()[0]), logfile, silence)
        printlog( "\t  : anneal_epochs : {:}".format(config.model.swa_anneal_epochs), logfile, silence)
        printlog( "\t  : anneal_strategy : '{:}'".format(swa_scheduler.anneal_func.__name__), logfile, silence)
    
    else:
        swa_scheduler = None

    return swa_scheduler



def get_swa_model(config:dict, model:nn.Module, optimizer:optim.Optimizer) -> Tuple[AveragedModel, SWALR]:
    """ configure model averaging scheme """
    logfile = config.out.log
    
    ##____ SWA model
    swa_model = AveragedModel(model)
    
    ##____ SWA scheduler
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=config.model.swa_lr,
        anneal_epochs=config.model.swa_anneal_epochs,
        anneal_strategy='cos',
    )
    
    return swa_model, swa_scheduler



def train_one_epoch(config:dict, model:Union[SVmodel, DDP], 
                    optimizer:optim.Optimizer, scheduler:LrScheduler, gradscaler:GradScaler, train_loader:DataLoader, margin_schedule:numpy.array=None) -> Tuple[dict, torch.Tensor]:
    """ model training for 1 epoch 

    Returns:
        train_scores (dict): train result stats
        running_mean_vec (Tensor): Size(embed_size,) - total mean of embedding vectors on training
    """
    logfile = config.out.log
    silence = not config.model.verbose
    
    start_t = time.time()
    grad_acc = config.model.grad_acc
    max_norm = config.model.max_norm
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    model_context = model.module if dist.is_initialized() else model
    
    total_loss       = 0.
    total_pred       = torch.empty(0)
    total_label      = torch.empty(0)
    running_mean_vec = torch.zeros(model_context.embed_size)
    corpus_size      = 0

    ##____ training iteration
    for i, (batch_waveform, batch_length, batch_label) in enumerate(tqdm(train_loader)):
        """ loader outputs the total batch size -> (e.g.) 512 batch_size * 2 grad_acc = 1024
        
        batch_waveform: Tensor (B, max_length)
        batch_length: Tensor (B,) - int
        batch_label: Tensor (B,) - int
        """
        ##____ escape conditions
        if config.args.quick_check and (i > 200) or (i+1)==len(train_loader): # drop last batch
            if dist.is_initialized(): dist.barrier()
            break
        
        ##____ apply margin schedule
        if margin_schedule is not None:
            m, mp, topK = model_context.aam_softmax.update_margins(margin=margin_schedule[i])
            if i==0: printlog("\t> margin: {:0.02f} | penalty: {:0.02f} | topK applied: {:d}".format(m, mp, topK), logfile, silence)
        
        ##____ split total batch input to chunks
        for chunk_waveform, chunk_length, chunk_label in zip(batch_waveform.chunk(grad_acc), batch_length.chunk(grad_acc), batch_label.chunk(grad_acc)):
            """ b = B // grad_acc 
            
            batch_waveform: Tensor (b, max_length)
            batch_length: Tensor (b,) - int
            batch_label: Tensor (b,) - int
            """
            ##____ model forward
            with autocast():
                # embed, pred, loss = model(x=chunk_waveform, target=chunk_label)
                embed, pred, loss = model(x=chunk_waveform.to(rank), target=chunk_label.to(rank))
                loss = loss / grad_acc
            
            ##____ backward & accum. training stats
            gradscaler.scale(loss).backward() # loss.backward()
            
            total_loss += loss.item()
            total_pred  = torch.cat([total_pred, pred.argmax(dim=-1).detach().cpu()], dim=0)
            total_label = torch.cat([total_label, chunk_label.detach().cpu()], dim=0)
            
            running_mean_vec += embed.detach().cpu().sum(dim=0) # (embed_size,) - accumulate the mini-batch mean vector
            corpus_size      += embed.size(0)

        ##____ update total batch gradients
        gradscaler.unscale_(optimizer)
        clip_grad_norm_(model_context.parameters(), max_norm)
        
        gradscaler.step(optimizer) # optim.step()
        gradscaler.update()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
    
    ##____ end of iteration, if DDP: rank==0 gather training stats from subproc's
    if dist.is_initialized():
        torch.save((total_loss, total_pred, total_label, running_mean_vec, corpus_size), opj(config.out.dir, f"part_result_gpu{config.gpus[rank]}_rank{rank}.pt"))
        dist.barrier()
        
        if rank==0:
            total_loss       = 0.
            total_pred       = torch.empty(0)
            total_label      = torch.empty(0)
            running_mean_vec = torch.zeros(model_context.embed_size)
            corpus_size      = 0
            
            for rank_id, gpu_id in enumerate(config.gpus):
                (part_loss, part_pred, part_label, part_mean_vec, part_size) = torch.load(opj(config.out.dir, f"part_result_gpu{gpu_id}_rank{rank_id}.pt"))
                total_loss  += part_loss
                total_pred  = torch.cat([total_pred,  part_pred], dim=0)
                total_label = torch.cat([total_label, part_label], dim=0)
                running_mean_vec += part_mean_vec
                corpus_size += part_size
                os.remove(opj(config.out.dir, f"part_result_gpu{gpu_id}_rank{rank_id}.pt"))
    
    ##____ train running mean vector
    running_mean_vec = running_mean_vec / corpus_size
        
    ##____ calculate training metrics
    train_scores = {'time': None, 'loss': 9999., 'ACC': 0., 'F1': 0.}
    if rank==0:
        train_scores['loss'] = total_loss / i
        train_scores['ACC'] = get_Accuracy(total_label, total_pred) * 100. # get percentile (%) units
        train_scores['F1']  = get_F1score(total_label, total_pred)  * 100.
    
    ##____ end of 1 epoch training
    if dist.is_initialized(): dist.barrier()
    torch.cuda.empty_cache()
    train_scores['time'] = time.time() - start_t
    
    return train_scores, running_mean_vec



def sync_model_weights(model:Union[SVmodel, DDP]) -> Union[SVmodel, DDP]:
    """ sync model weights before evaluation """
    model_context = model.module if dist.is_initialized() else model
    
    if dist.is_initialized():
        for p in model_context.parameters():
            dist.broadcast(p.data, src=0)
    
    model_state_dict = model_context.state_dict()
    for name, p in model_state_dict.items(): model_state_dict[name] = p.detach().clone() # to leaf tensor
    
    if dist.is_initialized(): dist.barrier()
    
    return model, model_state_dict
    

def sync_swa_model_weights(model:Union[SVmodel, DDP], swa_model:AveragedModel) -> Union[SVmodel, DDP]:
    """ sync model weights before evaluation """
    model_context = model.module if dist.is_initialized() else model
    
    ##____ broadcast swa model weights    
    if dist.is_initialized():
        for p in swa_model.parameters():
            dist.broadcast(p.data, src=0)
    
    ##____ load state from swa_model
    swa_state_dict = swa_model.module.state_dict()
    for name, p in swa_state_dict.items(): swa_state_dict[name] = p.detach().clone() # leaf tensor
    model_context.load_state_dict(swa_state_dict, strict=True)
    
    if dist.is_initialized(): dist.barrier()
    
    return model, swa_state_dict


def update_bn_ddp(config:dict, train_loader:DataLoader, swa_model:AveragedModel):
    """ update batch normalization factors at SWA model """
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    
    swa_model = swa_model.cuda(rank)
    if dist.is_initialized():
        swa_model = nn.SyncBatchNorm.convert_sync_batchnorm(swa_model)
        swa_model = DDP(module=swa_model, device_ids=[rank], find_unused_parameters=True)
        model_context = swa_model.module.module
    else: model_context = swa_model.module

    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum
    
    if not momenta: ##____ no batch normalization in model
        if dist.is_initialized(): swa_model = swa_model.module
        return swa_model
    
    for module in momenta.keys():
        module.momentum = None
    
    swa_model.train()
    running_mean_vec = torch.zeros(model_context.embed_size)
    corpus_size      = 0
    with torch.no_grad():
        for i, (batch_waveform, batch_length, batch_label) in enumerate(tqdm(train_loader)):
            
            ##____ escape condition
            if config.args.quick_check and (i > 300):
                if dist.is_initialized(): dist.barrier()
                break
            
            embed, _, _ = swa_model(x=batch_waveform.to(rank), target=batch_label.to(rank))
            
            ##____ accumulate mean vector stats
            running_mean_vec += embed.detach().cpu().sum(dim=0)
            corpus_size      += embed.size(0)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    
    swa_model.eval()
    if dist.is_initialized(): 
        torch.save((running_mean_vec, corpus_size), opj(config.out.dir, f"part_result_gpu{config.gpus[rank]}_rank{rank}.pt"))
        dist.barrier()
        
        if rank==0:
            running_mean_vec = torch.zeros(model_context.embed_size)
            corpus_size      = 0
            
            for rank_id, gpu_id in enumerate(config.gpus):
                (part_mean_vec, part_size) = torch.load(opj(config.out.dir, f"part_result_gpu{gpu_id}_rank{rank_id}.pt"))
                running_mean_vec += part_mean_vec
                corpus_size += part_size
                os.remove(opj(config.out.dir, f"part_result_gpu{gpu_id}_rank{rank_id}.pt"))
        
        swa_model = swa_model.module

    ##____ running mean vector
    running_mean_vec = running_mean_vec / corpus_size
   
    return swa_model, running_mean_vec



def save_checkpoint(config:dict, epoch:int, eval_scores:dict, 
                    optimizer_state:dict, scheduler_state:dict, model_state:dict, evaluation_corpus_embeddings:dict, running_mean_vec:torch.Tensor) -> None:
    """ save current checkpoint """
    
    ##____ clear checkpoint directory
    result_dir = opj(config.out.dir, f'{epoch}epoch')
    os.makedirs(result_dir)
    
    ##____ model state and metainfo
    # Path(opj(config.out.dir, 'checkpoint', '{:d}_epoch'.format(epoch))).touch()
    torch.save(epoch,           opj(result_dir, 'epoch.int'))
    torch.save(eval_scores,     opj(result_dir, 'eval_scores.dict'))
    torch.save(optimizer_state, opj(result_dir, 'optimizer.state'))
    torch.save(scheduler_state, opj(result_dir, 'scheduler.state'))
    torch.save(model_state,     opj(result_dir, 'model.state'))
    
    ##____ embedding outputs
    if not config.args.quick_check:
        torch.save(evaluation_corpus_embeddings, opj(result_dir, 'eval_corpus.embeddings'))
        torch.save(running_mean_vec,             opj(result_dir, 'running_mean_vec.tensor'))
    
    return


def update_bestshot(config:dict, best_scores:dict, eval_scores:dict, best_tracking:list, current_epoch:int) -> dict:
    """ update the best shot and keep top-N checkpoints """
    
    keep_nbest = config.model.keep_nbest
    
    ##____ best_tracking
    prev_best_epoch, prev_best_eer = best_tracking[0] if len(best_tracking) > 0 else (None, None)
    
    if config.args.quick_check:
        best_tracking.append((current_epoch, eval_scores['vox-O/EER']))
    else:
        best_tracking.append((current_epoch, numpy.mean([eval_scores['vox-O/EER'], eval_scores['vox-E/EER'], eval_scores['vox-H/EER']])))
    best_tracking.sort(key=lambda x: x[1])
    with open(opj(config.out.dir, 'best_tracking.txt'), 'w') as f:
        for (epoch, eer) in best_tracking: f.write(f'{epoch}, {eer:.04f}%\n')
    
    curr_best_epoch, curr_best_eer = best_tracking[0]
    
    ##____ best shot updated
    if prev_best_epoch != curr_best_epoch:
        best_scores = eval_scores
        
    ##____ keep top-N's, purge the rest
    for (remove_epoch, _) in best_tracking[keep_nbest:]:
        shutil.rmtree(opj(config.out.dir, f'{remove_epoch}epoch'), ignore_errors=True)
    
    return best_scores
    


def check_best_scores(best_scores:dict, eval_scores:dict):
    """ 
    Return:
        boolean to update the best or not
    """
    
    if len(best_scores)<=0: return True
    else:
        if best_scores['vox-O/EER'] > eval_scores['vox-O/EER']:
            return True
    
    return False



def log_results(config:dict, logger:Union[neptune.Run, None], current_epoch:int, train_scores:dict, eval_scores:dict, best_scores:dict):
    """ log model evaluation results """
    logfile = config.out.log
    
    ##____ neptune
    if logger is not None and config.args.neptune:
        for k in train_scores: logger[f"train/running/{k}"].append(train_scores[k], step=current_epoch)
        for k in eval_scores:
            if k=='time': continue
            logger[f"eval/running/{k}"].append(eval_scores[k], step=current_epoch)
        for k in best_scores:
            if k=='time': continue
            logger[f'eval/best/{k}'].append(best_scores[k], step=current_epoch)
        logger['eval/running/time'].append(eval_scores['time'], step=current_epoch)
    
    ##____ local loggings
    if len(train_scores): # train scores
        printlog("\t> 'train' | time: {:d}m {:02d}s".format(int(train_scores['time']//60), int(train_scores['time']%60)), logfile)
        printlog("\t>         | ACC: {:.02f}%, F1-macro: {:.02f}%, loss: {:.04f}".format(train_scores['ACC'], train_scores['F1'], train_scores['loss']), logfile)
        printlog("\t> ______________________________________________________", logfile)
        
    ##____ evaluation results
    printlog("\t> 'eval'  | time: {:d}m {:02d}s".format(int(eval_scores['time']//60), int(eval_scores['time']%60)), logfile)
    for trial_mode in ['vox-O', 'vox-E', 'vox-H']:
        ##____ escape condition
        if config.args.quick_check and trial_mode in ['vox-E', 'vox-H']:
            continue
        printlog("\t>          ---------------------------------------------", logfile)
        printlog("\t>  {:s}  | EER  : {:.03f}%".format(trial_mode, eval_scores[f'{trial_mode}/EER']), logfile)
        printlog("\t>         | DCF01: {:.04f}, threshold: {:.04f}".format(eval_scores[f'{trial_mode}/DCF01'], eval_scores[f'{trial_mode}/DCF01_thresh']), logfile)
        printlog("\t>         | DCF05: {:.04f}, threshold: {:.04f}".format(eval_scores[f'{trial_mode}/DCF05'], eval_scores[f'{trial_mode}/DCF05_thresh']), logfile)
    
    return
    