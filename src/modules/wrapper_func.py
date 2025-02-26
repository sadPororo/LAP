import torch.nn as nn

from transformers import Wav2Vec2Model, HubertModel, WavLMModel, Wav2Vec2Config, HubertConfig, WavLMConfig
from modules.pooling_layers.layer_aggregation import SUPERB, LAP
from modules.speaker_networks.astp import ASTP
from modules.speaker_networks.ecapa_tdnn import ECAPA_TDNN
from modules.speaker_networks.x_vector import X_vector
from modules.speaker_networks.mhfa import MHFA
from typing import Union



def frontend_module_selection(huggingface_cfg_name:str) -> Union[Wav2Vec2Model, HubertModel, WavLMModel]:
    """ frontend wrapper function
    
    Input:
        huggingface_cfg_name: choices['facebook/wav2vec2-base', 'facebook/wav2vec2-large', 'microsoft/wavlm-base', 'microsoft/wavlm-large', ...]
    Returns:
        Wave-pretrained models
    """
    if 'wav2vec2' in huggingface_cfg_name:
        return Wav2Vec2Model.from_pretrained(huggingface_cfg_name)
    elif 'hubert' in huggingface_cfg_name:
        return Wav2Vec2Model.from_pretrained(huggingface_cfg_name)
    elif 'wavlm' in huggingface_cfg_name:
        return WavLMModel.from_pretrained(huggingface_cfg_name)
    else:
        raise NotImplementedError(huggingface_cfg_name)



def backend_module_selection(config:dict, frontend_config:Union[Wav2Vec2Config, HubertConfig, WavLMConfig]) -> nn.ModuleDict:
    """ backend wrapper function """
    
    backend = nn.ModuleDict()
    
    ##____ MHFA has LAP/SUPERB as in-class component
    if config.model.speaker_network == 'mhfa':
        backend['speaker_network'] = MHFA(config, frontend_config)
        return backend
    
    
    ##___ layer aggregation
    if config.model.layer_aggregation == 'superb':
        backend['layer_aggregation'] = SUPERB(n_layer=frontend_config.num_hidden_layers+1)
        
    elif config.model.layer_aggregation == 'lap':
        backend['layer_aggregation'] = LAP(n_layer=frontend_config.num_hidden_layers+1, 
                                           size_in=frontend_config.hidden_size, 
                                           size_out=config.model.hidden_size, 
                                           n_head=config.model.n_head, 
                                           dropout=config.model.dropout)
    else:   raise NotImplementedError(f'layer_aggregation: {config.model.layer_aggregation}')
    
    
    ##____ speaker network
    if config.model.speaker_network=='astp':
        backend['speaker_network'] = ASTP(hidden_size=frontend_config.hidden_size if config.model.layer_aggregation=='superb' else config.model.hidden_size, 
                                          embed_dim=config.model.embed_size,
                                          reduction=0.5,
                                          dropout=config.model.dropout)
        
    elif config.model.speaker_network=='x-vector':
        backend['speaker_network'] = X_vector(in_features=frontend_config.hidden_size if config.model.layer_aggregation=='superb' else config.model.hidden_size, 
                                              hidden_size=512, 
                                              embed_size=512)
        
    elif config.model.speaker_network=='ecapa-tdnn':
        backend['speaker_network'] = ECAPA_TDNN(feat_dim=frontend_config.hidden_size if config.model.layer_aggregation=='superb' else config.model.hidden_size, 
                                                channels=512, embed_dim=192, global_context_att=True, emb_bn=False)
    else:   raise NotImplementedError(f'speaker_network: {config.model.speaker_network}')
    
    return backend


