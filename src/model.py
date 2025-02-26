#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from modules.wrapper_func import frontend_module_selection, backend_module_selection
from loss import AAMsoftmax_IntertopK_Subcenter


def init_weights(m:nn.Module):
    for name, p in m.named_parameters():
        if 'weight' in name and len(p.size()) > 1: 
            torch.nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            torch.nn.init.zeros_(p)


class SVmodel(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        
        ##____ speech-pretrained models
        self.frontend = frontend_module_selection(config.model.frontend_cfg)
        
        ##____ backend speaker model
        self.backend = backend_module_selection(config, self.frontend.config)
        init_weights(self.backend)
        self.embed_size = self.backend['speaker_network'].get_out_dim()
                
        ##____ training loss function                
        self.aam_softmax = AAMsoftmax_IntertopK_Subcenter(
            in_dim         = self.embed_size,
            n_class        = config.model.n_train_class * len(set(config.model.speed_perturb)),
            n_subcenter    = config.model.n_subcenter,
            margin         = config.model.margin, 
            scale          = config.model.scale,
            topK           = config.model.topK,
            margin_penalty = config.model.margin_penalty
        )
        
    def freeze_frontend_parameters(self):
        """ freeze frontend model """
        for p in self.frontend.parameters(): p.requires_grad_(False)
        self.frontend.eval()
        
    def update_frontend_parameters(self):
        """ unfreeze frontend parameters """
        for p in self.frontend.parameters(): p.requires_grad_(True)
        self.frontend.train()

    def detach(self):
        """ detach all model parameters """
        for p in self.parameters(): p.data = p.data.clone().detach()
        for p in self.buffers():    p.data = p.data.clone().detach()
        for name, module in self.named_modules():
            for attr_name, attr_val in vars(module).items():
                if isinstance(attr_val, torch.Tensor): setattr(module, attr_name, attr_val.clone().detach())
        self.load_state_dict({name: p.clone().detach() for name, p in self.state_dict().items()})

    def forward(self, x:torch.Tensor, length:torch.LongTensor=None, target:torch.LongTensor=None):
        """ 
        Inputs: 
            x: (B T); batch_size, waveform_length
            length: (B)
            target: (B)
            
        Returns:
            if training:
                x: (B E); batch_size, speaker_embeding_size
                cls_pred: (B nClass); batch_size, train_speaker_num
                loss: (1); batch_loss            
            else:
                x: (B E)
        """
        ##____ attention mask for frontend module
        if length is not None:
            length = self.frontend._get_feat_extract_output_lengths(length)
            attn_mask = torch.arange(length.max().item(), device=x.device) < length[:, None]
        else:   attn_mask = None
        
        ##____ frontend forward
        x = self.frontend.forward(x, attention_mask=attn_mask, output_hidden_states=True)
        x = torch.stack(x.hidden_states) # (L B T D)
        x = rearrange(x, 'L B T D -> B D T L')
        
        ##____ backend forward
        for layer_nm in self.backend:
            x = self.backend[layer_nm](x)
        
        ##____ aam-softmax
        if self.training:
            cls_pred, loss = self.aam_softmax(x, target)
            return x, cls_pred, loss
        else:
            return x


# %%
if __name__ == '__main__':

    from easydict import EasyDict
    from collections import OrderedDict
    #%%
    from utils.utility import hypload
    # from ptflops import get_model_complexity_info
    from tqdm import tqdm

    from os.path import join as opj
    import numpy as np
    # import time
    
    config = EasyDict()
    config['model'] = hypload('./config/train_frozen.yaml')
    
    model = SVmodel(config)

    #%%
    model_state = torch.load(opj('/home/jinsob/UniPool-Ext/res/LAP-454/3epoch/model.state'), map_location='cpu')
        
    key_mapping = {
         'backend.layer_aggregation.norm.weight': 'backend.layer_aggregation.in_norm.weight',
         'backend.layer_aggregation.norm.bias': 'backend.layer_aggregation.in_norm.bias',
         'backend.layer_aggregation.norm.running_mean': 'backend.layer_aggregation.in_norm.running_mean',
         'backend.layer_aggregation.norm.running_var': 'backend.layer_aggregation.in_norm.running_var',
         'backend.layer_aggregation.norm.num_batches_tracked': 'backend.layer_aggregation.in_norm.num_batches_tracked',
         'backend.layer_aggregation.projection.weight': 'backend.layer_aggregation.in_weight.weight',
         'backend.layer_aggregation.projection.bias': 'backend.layer_aggregation.in_weight.bias',
         'backend.speaker_network.linear1.weight': 'backend.layer_aggregation.out_weight.weight',
         'backend.speaker_network.linear1.bias': 'backend.layer_aggregation.out_weight.bias',
         'backend.speaker_network.norm1.weight': 'backend.layer_aggregation.out_norm.weight',
         'backend.speaker_network.norm1.bias': 'backend.layer_aggregation.out_norm.bias',
         'backend.speaker_network.norm1.running_mean': 'backend.layer_aggregation.out_norm.running_mean',
         'backend.speaker_network.norm1.running_var': 'backend.layer_aggregation.out_norm.running_var',
         'backend.speaker_network.norm1.num_batches_tracked': 'backend.layer_aggregation.out_norm.num_batches_tracked',

         'backend.speaker_network.linear2.weight' : 'backend.speaker_network.emb_weight.weight',
         'backend.speaker_network.linear2.bias' : 'backend.speaker_network.emb_weight.bias',
         'backend.speaker_network.norm2.weight' : 'backend.speaker_network.emb_norm.weight',
         'backend.speaker_network.norm2.bias' : 'backend.speaker_network.emb_norm.bias',
         'backend.speaker_network.norm2.running_mean' : 'backend.speaker_network.emb_norm.running_mean',
         'backend.speaker_network.norm2.running_var' : 'backend.speaker_network.emb_norm.running_var',
         'backend.speaker_network.norm2.num_batches_tracked' : 'backend.speaker_network.emb_norm.num_batches_tracked'
    }
    
    new_model_state = OrderedDict(
        (key_mapping[k] if k in key_mapping else k, v) for k, v in model_state.items()
    )
    #%%
    
    model.load_state_dict(new_model_state, strict=True)
    
    #%%
    
    torch.save(new_model_state ,'/home/jinsob/UniPool-Ext/res/LAP-ASTP-LMFT/3epoch/model.state')
    
# %%
