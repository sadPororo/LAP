import torch
import torch.nn as nn

# from transformers import Wav2Vec2Model, HubertModel, WavLMModel
from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig
from modules.pooling_layers.layer_aggregation import SUPERB, LAP
from typing import Union


class MHFA(nn.Module):
    """ 
    Reference:
        An attention-based backend allowing efficient fine-tuning of transformer models for speaker verification (2022, SLT)

    Modified from the code available at:
        https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling
    """
    def __init__(self, config:dict, frontend_config:Union[Wav2Vec2Config, HubertConfig, WavLMConfig],
                 head_nb=64, compression_dim=128, outputs_dim=256):
        super().__init__()

        # Define learnable weights for key and value computations across layers
        if config.model.layer_aggregation == 'superb':
            self.l_pool_k = SUPERB(frontend_config.num_hidden_layers+1)
            self.l_pool_v = SUPERB(frontend_config.num_hidden_layers+1)
        elif config.model.layer_aggregation == 'lap':
            self.l_pool_k = LAP(n_layer=frontend_config.num_hidden_layers+1, size_in=frontend_config.hidden_size, size_out=config.model.hidden_size, n_head=config.model.n_head, dropout=config.model.dropout)
            self.l_pool_v = LAP(n_layer=frontend_config.num_hidden_layers+1, size_in=frontend_config.hidden_size, size_out=config.model.hidden_size, n_head=config.model.n_head, dropout=config.model.dropout)
        else:   raise NotImplementedError(f'layer_aggregation: {config.model.layer_aggregation}')
        # self.weights_k = nn.Parameter(data=torch.ones(layer_nb), requires_grad=True)
        # self.weights_v = nn.Parameter(data=torch.ones(layer_nb), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.layr_nb = frontend_config.num_hidden_layers + 1
        self.ins_dim = frontend_config.hidden_size if config.model.layer_aggregation=='superb' else config.model.hidden_size        
        # self.layr_nb = layer_nb
        # self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)
    
    def get_out_dim(self):
        return self.pooling_fc.out_features

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = self.l_pool_k(x).transpose(1, 2) # B C T -> B T C
        # k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2) # B C T -> B T C

        # Compute the value in a similar fashion
        v = self.l_pool_v(x).transpose(1, 2) # B C T -> B T C
        # v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2) # B C T -> B T C

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k) # B T C'
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k) # B T C' -> B T H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B T 1 C'

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1) # sum(v:(B T 1 C') matmul s:(B prob H, 1)) dim=T)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs        
