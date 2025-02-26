
import torch
import torch.nn as nn
import torch.nn.functional as F



class Attentive_Statistc_Pooling(nn.Module):
    """ 
    Reference:
        ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification (2021, Interspeech)
    """
    def __init__(self, size_in:int, reduction:float=0.5, dropout:float=0.1, normout:bool=True):
        super().__init__()
        self.size_in = size_in
        
        ##____ channel-dependent attention scoring mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(size_in*3, int(size_in*reduction), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(size_in*reduction)),
            nn.Conv1d(int(size_in*reduction), size_in, kernel_size=1),
            )
        self.p_dropout = dropout
        self.norm = nn.BatchNorm1d(size_in*2) if normout else nn.Identity()
    
    def get_out_dim(self):
        return self.size_in * 2
                
    def forward(self, x:torch.Tensor, mask=None):
        """ 
        Input
            x: (B C T) - float
            mask: (B T) - bool
        Output
            x: (B 2C)
        """
        B, C, T = x.size()
        
        ##____ calculate time-dependent statistics
        if mask is not None:
            mask = mask[:, None, :] # (B 1 T)
            N    = mask.sum(dim=-1, keepdim=True) # (B 1 1)
            mu = (x * mask).sum(dim=-1, keepdim=True) / N # (B C 1)
            sg = torch.sqrt((((x - mu) ** 2) * mask).sum(dim=-1, keepdim=True) / N)
        else:
            mask = torch.ones(B, 1, T, dtype=bool, device=x.device)
            mu = x.mean(dim=-1, keepdim=True) # (B C 1)
            sg = x.std(dim=-1, keepdim=True)

        stat_pool = torch.cat([x, mu.expand(-1,-1,T), sg.expand(-1,-1,T)], dim=1) # (B 3C T)
        
        ##____ channel-dependent attention scoring
        attn_scr = self.attention(stat_pool) # (B C T)
        if self.training: # score dropout
            mask = mask & ~(torch.rand(mask.size(), device=x.device) < self.p_dropout)
        attn_scr.masked_fill_(~mask, torch.finfo(attn_scr.dtype).min) # mask: (B 1 T)
        attn_scr = F.softmax(attn_scr, dim=-1) # (B C T)
        
        ##____ get attentive statistics
        attn_mu = torch.sum(x * attn_scr, dim=-1) # (B C)
        attn_sg = torch.sqrt((torch.sum((x**2) * attn_scr, dim=-1) - attn_mu**2).clamp(min=1e-4)) # (B C)
        attn_pool = torch.cat([attn_mu, attn_sg], dim=1) # (B 2C)
        attn_pool = self.norm(attn_pool)
        
        return attn_pool, attn_scr



class ASTP(nn.Module):
    """ 
        Temporal pooling and embedding projection layer
    """
    def __init__(self, hidden_size=512, embed_dim=192, reduction=0.5, dropout=0.1, emb_bn=True):
        """ 
        """
        super().__init__()
        
        ##____ attentive stat pooling
        self.t_pool  = Attentive_Statistc_Pooling(hidden_size, reduction=reduction, dropout=dropout)
        pool_out_dim = self.t_pool.get_out_dim()
        
        ##____ projection for input shape (B C) 
        self.emb_weight = nn.Linear(pool_out_dim, embed_dim)
        self.emb_norm   = nn.BatchNorm1d(embed_dim) if emb_bn else nn.Identity()

    def get_out_dim(self):
        return self.emb_weight.out_features
    
    def forward(self, x:torch.Tensor, output_attention:bool=False):
        """ Input 
                x: (B D1 T) 
            Output
                x: (B D2)
        """
        
        x, attn_scr = self.t_pool(x) # (B D)
        x = self.emb_norm(self.emb_weight(x))
                
        if output_attention:
            return x, attn_scr
        else:
            return x