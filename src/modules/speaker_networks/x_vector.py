import torch
import torch.nn as nn
import modules.pooling_layers.temporal_aggregation as temporal_pooling

class X_vector(nn.Module):
    """ 
    Reference: 
        X-vectors: Robust dnn embeddings for speaker recognition (2018, ICASSP)
    """
    def __init__(self, in_features:int, hidden_size:int=512, out_features:int=1500, embed_size:int=512):
        super().__init__()
        
        in_channel_list    = [in_features] + [hidden_size] * 4
        out_channel_list   = [hidden_size] * 4 + [out_features]
        kernel_size_list   = [5, 3, 3, 1, 1]
        padding_size_list  = [2, 1, 1, 0, 0]
        dilation_size_list = [1, 2, 3, 1, 1]
        
        ##____ tdnn
        self.layers = nn.ModuleList()
        for i in range(len(kernel_size_list)):
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_channel_list[i],
                    out_channels=out_channel_list[i],
                    kernel_size=kernel_size_list[i],
                    dilation=dilation_size_list[i],
                    padding=padding_size_list[i]
                )
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(out_channel_list[i]))
        
        ##____ statistic (mean, std) pooing
        self.pool = getattr(temporal_pooling, 'TSTP')(in_dim=out_features)
        # self.pool = getattr(pooling_layers, 'TSTP')(in_dim=out_features)
        
        ##____ projection
        self.linear1 = nn.Linear(out_features*2, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.relu    = nn.ReLU()
    
    def get_out_dim(self):
        return self.linear1.out_features
    
    def forward(self, x:torch.Tensor):
        """ x : (B D T) """
        
        ##____ tdnn
        for layer in self.layers:
            x = layer(x)
        
        ##____ stat pooling
        x = self.pool(x)
        
        ##____ projection
        if self.training:
            return self.linear2(self.relu(self.linear1(x)))
        else:
            return self.linear1(x)
            

#%%
if __name__ == '__main__':
    
    x = torch.zeros(1, 1024, 99)
    model = X_vector(in_features=1024)
    model.eval()
    out = model(x)
    print(out.shape)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 1024, 199)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))

# %%
