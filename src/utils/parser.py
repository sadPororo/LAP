import os

from ast import literal_eval
from os.path import isdir
from utils.utility import route_dicthierarchy, hypload, setInDict
from utils.loggers import printlog
from easydict import EasyDict


def update_argument(hyps, kwargs=None):
    """ update values in configurations given '--kwargs' arguments """
    
    if kwargs == None:
        return hyps
    
    for k in hyps.keys():
        if k in kwargs.keys():
            print(f"> target keyword hyp spotted, '{k}': {hyps[k]} >> {kwargs[k]}")
            hyps[k] = type(hyps[k])(kwargs[k])
                
    return hyps


def get_configurations(args, fpath:str):
    """ load yaml configuration file """
    
    ##____ setup to adjust settings
    kwarg_lines = [] if args.kwargs is None else [i.strip() for i in args.kwargs.split('--')] # '--max_lr 0.01 --load_pretrained' -> ['max_lr 0.01', 'load_pretrained']
    kwargs = {}
    for i in kwarg_lines: # -> {'max_lr': '0.01', 'load_pretrained': True}
        if ' ' in i: kwargs[i.split(' ')[0]] = i.split(' ')[1]
        else: kwargs[i] = True 
    
    ##____ load .yaml configuration file
    config = EasyDict()
    config.args = dict(vars(args))
    config.gpus = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
    config.model = update_argument(hypload(fpath), kwargs)
    config.neptune = update_argument(hypload('./config/neptune.yaml'))
    
    return config
    

    

    
    