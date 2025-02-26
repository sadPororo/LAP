""" Main function to initiate the experiment """

import os
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.parser import get_configurations
from utils.utility import random_state_init, ddp_setup
from utils.loggers import init_loggers, printlog

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description = 'main function to initiate pipeline')
parser.add_argument('--train_frozen',     dest='train_frozen',     action='store_true') # 1-st phase
parser.add_argument('--train_finetune',   dest='train_finetune',   action='store_true') # 2
parser.add_argument('--train_lmft',       dest='train_lmft',       action='store_true') # 3
parser.add_argument('--naive_evaluation', dest='naive_evaluation', action='store_true') # 4
parser.add_argument('--score_normalize',  dest='score_normalize',  action='store_true') # 5
parser.add_argument('--score_calibrate',  dest='score_calibrate',  action='store_true') # 6
parser.add_argument('--quick_check',      dest='quick_check',      action='store_true')
parser.add_argument('--neptune',          dest='neptune',          action='store_true')
parser.add_argument('--description',      default='', type=str)
parser.add_argument('--evaluation_id',    default='', type=str)
parser.add_argument('--kwargs',           default='', type=str)
args = parser.parse_args()




def ddp_main(rank:int, config:dict, phase:int):
    torch.cuda.set_device(rank)
    
    ##____ initiate torch distributed
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=len(config.gpus),
        rank=rank
    )
        
    ##____ start logger
    config, logger = init_loggers(config, phase, rank==0)
    printlog(f"GPU device({config.gpus[rank]}) ready on rank {rank}.", config.out.log, forced_log=True)
    ddp_setup(rank==0)
    dist.barrier()

    ##____ phase control :: train | evaluation
    if phase in [1,2,3]:
        if phase==1:    from train.frozen_front   import train_model
        elif phase==2:  from train.joint_finetune import train_model
        elif phase==3:  from train.joint_lmft     import train_model
        random_state_init(seed=config.model.seed)
        train_model(config, logger)
    
    elif phase in [4,5,6]:
        if phase==4:    from evaluate.naive_score     import evaluate_model
        elif phase==5:  from evaluate.norm_score      import evaluate_model
        elif phase==6:  
            from evaluate.qmf_calibration import evaluate_model
            random_state_init(seed=config.model.seed)
        evaluate_model(config, logger)
    
    ##____ pass the experiment metainfo to the main function
    if rank==0:
        torch.save(config, '../tmp/config_exp_on_{:}.config'.format(''.join(config.gpus)))

    dist.destroy_process_group()




def single_main(config:dict, phase:int):
    
    ##____ start logger
    config, logger = init_loggers(config, phase, is_master=True)
    printlog(f"GPU device({config.gpus[0]}) ready.", config.out.log)
    
    ##____ phase control :: train | evaluation
    if phase in [1,2,3]:
        if phase==1:    from train.frozen_front   import train_model
        elif phase==2:  from train.joint_finetune import train_model
        elif phase==3:  from train.joint_lmft     import train_model
        random_state_init(seed=config.model.seed)
        train_model(config, logger)
    
    elif phase in [4,5,6]:
        if phase==4:    from evaluate.naive_score     import evaluate_model
        elif phase==5:  from evaluate.norm_score      import evaluate_model
        elif phase==6:  
            from evaluate.qmf_calibration import evaluate_model
            random_state_init(seed=config.model.seed)
        evaluate_model(config, logger)

    return config



def pass_check(config:dict, prev_id:str=None):
    """ pass the previous exp_id to the next step """
    
    ##____ command is not continous run
    if prev_id is None:
        assert args.evaluation_id != ''
        
    else: ##____ for the continuous exp-run
        config.args.evaluation_id = prev_id
        
    return config



def phase_manager(config:dict, phase:int):
    """ wrapper function """
    
    ##____ train::random state init
    if phase in [1,2,3,6]:
        if phase in [1,2,3]:
            config.phase = phase
        config.model.seed = config.model.seed * (config.model.seed % phase + 1) // phase
        random_state_init(seed=config.model.seed)
    
    ##____ DDP
    if len(config.gpus) > 1:
        mp.spawn(ddp_main, args=(config, phase), nprocs=len(config.gpus), join=True)
        config = torch.load('../tmp/config_exp_on_{:}.config'.format(''.join(config.gpus)))
        os.remove('../tmp/config_exp_on_{:}.config'.format(''.join(config.gpus)))
    
    else: # single GPU
        config = single_main(config, phase)
    
    ##____ return 'exp_id'
    return config.out.id



if __name__ == "__main__":
    prev_id = None
    
    if args.train_frozen:
        config = get_configurations(args, './config/train_frozen.yaml')
        prev_id = phase_manager(config, phase=1)
    
    if args.train_finetune:
        config = get_configurations(args, './config/train_joint_ft.yaml')
        config = pass_check(config, prev_id)
        prev_id = phase_manager(config, phase=2)
    
    if args.train_lmft:
        config = get_configurations(args, './config/train_joint_lmft.yaml')
        config = pass_check(config, prev_id)
        prev_id = phase_manager(config, phase=3)

    if args.naive_evaluation:
        config = get_configurations(args, './config/eval_basic.yaml')
        config = pass_check(config, prev_id)
        _ = phase_manager(config, phase=4)
    
    if args.score_normalize:
        config = get_configurations(args, './config/eval_norm_score.yaml')
        config = pass_check(config, prev_id)
        _ = phase_manager(config, phase=5)
    
    if args.score_calibrate:
        config = get_configurations(args, './config/eval_qmf_calibration.yaml')
        config = pass_check(config, prev_id)
        _ = phase_manager(config, phase=6)
    
   

