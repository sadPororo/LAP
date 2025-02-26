import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange




class SUPERB(nn.Module):
    """ 
    Reference:
        SUPERB: Speech processing Universal PERformance Benchmark (2021, Interspeech)
    """
    def __init__(self, n_layer:int):
        super().__init__()
        assert n_layer > 0
        
        ##____ softmax logits
        self.layer_weights = nn.Parameter(data=(torch.ones(n_layer)/n_layer), requires_grad=True)
        
    def forward(self, x:torch.Tensor):
        """ 
        Input:
            x: (B D T L); batch_size, hidden_size, time_length, n_layer
            
        Returns
            x: (B D T)
        """
        return (x * F.softmax(self.layer_weights, dim=-1)[None, None, None, :]).sum(dim=-1)



class LAP(nn.Module):
    def __init__(self, n_layer:int, size_in:int, size_out:int, n_head:int, dropout:float=0.1):
        super().__init__()
        
        assert size_out % n_head==0
        self.n_head = n_head

        ##____ W_in
        self.in_norm = nn.BatchNorm2d(size_in)
        self.in_weight = nn.Conv2d(size_in, size_in, kernel_size=(1, 1)) if n_head > 1 else nn.Identity()
        
        ##____ squeeze & expand
        self.sq_weight = nn.Parameter(torch.randn(n_head, n_layer, n_layer//2))
        self.sq_bias   = nn.Parameter(torch.zeros(n_head,       1, n_layer//2))
        self.ex_weight = nn.Parameter(torch.randn(n_head, n_layer//2, n_layer))
        self.ex_bias   = nn.Parameter(torch.zeros(n_head,          1, n_layer))
        
        ##____ score dropout
        self.dropout = nn.Dropout(p=dropout)
        
        ##____ W_out
        self.out_weight = nn.Conv1d(in_channels=size_in, out_channels=size_out, kernel_size=1)
        self.out_norm   = nn.BatchNorm1d(size_out)

    def forward(self, x:torch.Tensor, output_attention:bool=False):
        """ 
        Input:
            x: (B D1 T L); batch_size, hidden_size, time_length, n_layer
            
        Returns
            x: (B D2 T)
        """
        B = x.size(0)
        
        ##____ multi-head projection
        x = self.in_weight(self.in_norm(x)) # (B D T L)
        x = rearrange(x, 'B (h d) T L -> (B T) h d L', h=self.n_head)
        
        ##____ head-wise stats
        mu = x.mean(dim=2, keepdim=True) # ([BT] h 1 L)
        mx = x.max(dim=2, keepdim=True)[0]
        
        ##____ shared sq & ex attention
        mu = torch.matmul(F.relu(torch.matmul(mu, self.sq_weight) + self.sq_bias), self.ex_weight) + self.ex_bias # ([BT] h 1 L)
        mx = torch.matmul(F.relu(torch.matmul(mx, self.sq_weight) + self.sq_bias), self.ex_weight) + self.ex_bias
        s  = self.dropout(F.sigmoid(mu + mx)) # ([BT] h 1 L)
        x  = x * s # ([BT] h d L)

        ##____ pool out & head concat
        x = x.max(dim=-1)[0] # ([BT] h d)
        x = rearrange(x, '(B T) h d -> B (h d) T', B=B) # (B D T)
        
        ##____ out projection
        x = self.out_norm(self.out_weight(x)) # (B D T)
        
        if output_attention:
            s = rearrange(s, '(B T) h 1 L -> B h T L')
            return x, s
            
        else:
            return x
