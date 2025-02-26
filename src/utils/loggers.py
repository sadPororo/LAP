import os
import sys
import logging
import neptune

import torch.distributed as dist

from easydict import EasyDict
from datetime import datetime
from os.path import join as opj
from utils.utility import hypwrite, hypload



def printlog(stdout:str, logfile_path:str=None, silence:bool=False, forced_log:bool=False):
    """ Print the string object to sys.stdout and make a log
    
    Args:
        stdout (str): content for stdout.
        logfile_path (str): '.log' file path.
        silence (bool): will not print if given True but will make a log. Defaults to False.
        forced_log (bool): will force to make a log, even it's slave process. Defaults to False
    """
    if dist.is_initialized(): is_master = dist.get_rank()==0
    else: is_master = True
    
    if not silence:
        for line in stdout.replace('\t', ' '*4).split('\n'):
            print(line)
    
    # create a log with the same content (only the master process creates the log)
    if (is_master and logfile_path is not None) or forced_log:
        logging.basicConfig(filename=logfile_path,
                            format='%(asctime)s %(message)s',
                            datefmt='%Y/%m/%d (%I:%M:%S %p)',
                            filemode='a+')
        local_logger = logging.getLogger('train')
        local_logger.setLevel(logging.DEBUG)
        for line in stdout.split('\n'):
            local_logger.debug(line)


def init_loggers(config:dict, phase:int, is_master:bool=True):
    """ Initiate experiment loggers

    Args:
        config (dict): hyperparameters
        phase (int): current process phase.
        is_master (bool): rank==0. Defaults to True.
    """
    if is_master: ##____ only master process manage the loggers
        if config.args.neptune:
            tags = (
                config.neptune.location_tag                                 # machine location (IP, names)
                + [f"cuda: [{','.join(config.gpus)}]"]                      # GPU-device list
                + [os.path.abspath(os.getcwd()).split('/')[-1]]             # ./src version tag
                + [['frozen', 'finetune', 'lmft', 'naive_score', 'norm_score', 'qmf_calibration'][phase-1]] # current phase
            )

            logger = neptune.init_run(
                project = config.neptune.project,
                api_token = config.neptune.api_token,
                tags=tags,
                description=config.args.description                
            )
            logger['parameters'] = config
            exp_id = str(logger._sys_id)
            
            if phase in [1,2,3]: # if training phase, add self-id tag
                logger['sys/tags'].add(exp_id)
                            
            if phase in [2,3,4,5,6]: # while phase using previous experiment result, add the previous-id
                logger['sys/tags'].add(config.args.evaluation_id)
            
            if phase in [4,5,6]: # if evaluation phase, add tag of current testing phase
                prev_config = hypload(opj('../res', config.args.evaluation_id, 'config.yaml'))
                logger['sys/tags'].add(['frozen', 'finetune', 'lmft', 'naive_score', 'norm_score', 'qmf_calibration'][prev_config.phase-1])
        
        else:
            logger = None
            exp_id = 'local-' + datetime.now().strftime("%Y%m%d-%H%M%S")
            
        with open('../tmp/exp_id.txt', 'w') as f: 
            f.write(exp_id)

    ##____ if not a master, get exp_id from master
    if dist.is_initialized(): dist.barrier()
    if not is_master:
        logger = None
        with open('../tmp/exp_id.txt', 'r') as f: 
            exp_id = f.readline()
        exp_id = exp_id.strip()
    if dist.is_initialized(): dist.barrier()
    if is_master: os.remove('../tmp/exp_id.txt')
    
    ##____ set output paths
    config.out = {
        'id': exp_id,
        'dir': opj('../res', exp_id), 
        'log': opj('../res', exp_id, 'training.log')
    }
    os.makedirs(config.out.dir, exist_ok=True)
    
    ##____ make the first log
    if is_master:
        printlog('\nUSING COMMAND: CUDA_VISIBLE_DEVICES='+ os.environ.get('CUDA_VISIBLE_DEVICES') + ' python ' + ' '.join(sys.argv)+'\n',  config.out.log)
        printlog(f"Experiment '{exp_id}' logger created.", config.out.log)
        printlog(f"\t> result directory path: '{os.path.abspath(opj(os.getcwd(), config.out.dir))}/'\n", config.out.log)
        
        ##____log hyperparameters to result folder
        hypwrite(config, opj(config.out.dir, 'config.yaml'))

    if dist.is_initialized(): dist.barrier()
    
    return config, logger

