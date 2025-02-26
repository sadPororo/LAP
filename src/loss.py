import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
    
class AAMsoftmax_IntertopK_Subcenter(nn.Module):
    def __init__(self, in_dim:int, n_class:int,   # classifier
                 margin:float, scale:float,       # ArcFace
                 n_subcenter:int,                 # subcenter K
                 topK:int, margin_penalty:float): # InterTopK, penalty
        """
        code modified from: https://github.com/wenet-e2e/wespeaker/blob/
                            c9ec537b53fe1e04525be74b2550ee95bed3a891/wespeaker/models/projections.py#L243
        
        Reference:
            ArcFace: Additive Angular Margin Loss for Deep Face Recognition (2019, CVPR)
            Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces (2020, ECCV)
            Multi-Query Multi-Head Attention Pooling and Inter-Topk Penalty for Speaker Verification (2022, ICASSP)
        """        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = n_class
        
        self.m = margin
        self.scale = scale
        self.n_subcenter = n_subcenter

        # ArcFace setups
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

        # Inter-top K penalty setups
        self.topK = topK
        self.margin_penalty = margin_penalty
        self.cos_p = math.cos(0.)
        self.sin_p = math.sin(0.)
        
        # class projection and loss
        self.weight = nn.Parameter(torch.zeros(self.n_subcenter, self.in_dim, self.out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()
            
    def update_margins(self, margin:float, margin_penalty:float=None, topK:int=None):
        # update ArcFace elements
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin
        
        if margin_penalty is not None:
            self.margin_penalty = margin_penalty
        
        # rescale Inter-top K penalty margin
        if margin > 0.001:
            mp = self.margin_penalty * (margin / self.m)
        else:
            mp = 0.0
        self.cos_p = math.cos(mp)
        self.sin_p = math.sin(mp)
        
        # update topK
        if topK is not None:
            self.topK = topK
        
        return margin, mp, self.topK

    def get_current_margins(self):
        return math.asin(self.sin_m), math.asin(self.sin_p), self.topK
            
    def forward(self, x, label):
        """ 
            x : (B, D)
            label: (B)
        """
        # cos-similarity from each subcenter, then pool out the maximum
        cosine = torch.matmul(F.normalize(x, dim=1, p=2), F.normalize(self.weight, dim=1, p=2)) # (n_subcenter, B, n_class)
        cosine, _ = cosine.max(dim=0) # (B, n_class)
        
        sine = torch.sqrt( (1.0 - torch.pow(cosine, 2)) )
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)
                
        # one-hot label
        one_hot = torch.zeros_like(cosine) # (B, n_class)
        one_hot = one_hot.scatter_(-1, label[..., None], 1)
        
        # topK penalty
        if self.topK > 0:
            _, topK_indice = (cosine - 2 * one_hot).topk(self.topK, dim=1)
            topK_one_hot   = torch.zeros_like(cosine).scatter_(-1, topK_indice, 1)

            phi_penalty = cosine * self.cos_p + sine * self.sin_p # cos(theta_i,j - m')
            logit = (one_hot * phi) + (topK_one_hot * phi_penalty) + ((1.0 - one_hot - topK_one_hot) * cosine)
            
        else:
            logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            
        # Cross Entropy loss
        logit = logit * self.scale
        loss  = self.ce(logit, label)
        
        # speaker classification
        with torch.no_grad():
            pred = ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale
            pred = F.softmax(pred, dim=-1) # (B, n_class)
            
        return pred, loss
        
        
             
# #%%

# from espnet2.spk.loss.abs_loss import AbsLoss

# class ArcMarginProduct_intertopk_subcenter(AbsLoss):
#     r"""Implement of large margin arc distance with intertopk and subcenter:

#     Reference:
#         MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
#         FOR SPEAKER VERIFICATION.
#         https://arxiv.org/pdf/2110.05042.pdf
#         Sub-center ArcFace: Boosting Face Recognition by
#         Large-Scale Noisy Web Faces.
#         https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         scale: norm of input feature
#         margin: margin
#         cos(theta + margin)
#         K: number of sub-centers
#         k_top: number of hard samples
#         mp: margin penalty of hard samples
#         do_lm: whether do large margin finetune
#     """

#     def __init__(
#         self,
#         nout,
#         nclasses,
#         scale=32.0,
#         margin=0.2,
#         easy_margin=False,
#         K=3,
#         mp=0.06,
#         k_top=5,
#         do_lm=False,
#     ):
#         super().__init__(nout)
#         self.in_features = nout
#         self.out_features = nclasses
#         self.scale = scale
#         self.margin = margin
#         self.do_lm = do_lm

#         # intertopk + subcenter
#         self.K = K
#         if do_lm:  # if do LMF, remove hard sample penalty
#             self.mp = 0.0
#             self.k_top = 0
#         else:
#             self.mp = mp
#             self.k_top = k_top

#         # initial classifier
#         self.weight = nn.Parameter(torch.FloatTensor(self.K * nclasses, nout))
#         nn.init.xavier_uniform_(self.weight)

#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)
#         self.th = math.cos(math.pi - margin)
#         self.mm = math.sin(math.pi - margin) * margin
#         self.mmm = 1.0 + math.cos(
#             math.pi - margin
#         )  # this can make the output more continuous
#         ########
#         self.m = self.margin
#         ########
#         self.cos_mp = math.cos(0.0)
#         self.sin_mp = math.sin(0.0)

#         self.ce = nn.CrossEntropyLoss()

#     def update(self, margin=0.2):
#         self.margin = margin
#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)
#         self.th = math.cos(math.pi - margin)
#         self.mm = math.sin(math.pi - margin) * margin
#         self.m = self.margin
#         self.mmm = 1.0 + math.cos(math.pi - margin)

#         # hard sample margin is increasing as margin
#         if margin > 0.001:
#             mp = self.mp * (margin / 0.2)
#         else:
#             mp = 0.0
#         self.cos_mp = math.cos(mp)
#         self.sin_mp = math.sin(mp)

#     def forward(self, input, label):
#         if len(label.size()) == 2:
#             label = label.squeeze(1)
#         cosine = F.linear(
#             F.normalize(input), F.normalize(self.weight)
#         )  # (batch, out_dim * k)
#         cosine = torch.reshape(
#             cosine, (-1, self.out_features, self.K)
#         )  # (batch, out_dim, k)
#         cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         phi_mp = cosine * self.cos_mp + sine * self.sin_mp

#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             ########
#             # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#             phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
#             ########

#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, label.view(-1, 1), 1)

#         if self.k_top > 0:
#             # topk (j != y_i)
#             _, top_k_index = torch.topk(
#                 cosine - 2 * one_hot, self.k_top
#             )  # exclude j = y_i
#             top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

#             # sum
#             output = (
#                 (one_hot * phi)
#                 + (top_k_one_hot * phi_mp)
#                 + ((1.0 - one_hot - top_k_one_hot) * cosine)
#             )
#         else:
#             output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.scale

#         loss = self.ce(output, label)
#         return loss
    



# class Subcentered_AAMsoftmax(nn.Module):
#     def __init__(self, in_dim:int, n_class:int, n_subcenter:int, margin:float, scale:float, topK:int, penalty:float):
#         super().__init__()
        
#         self.in_dim = in_dim
#         self.n_class = n_class
#         self.n_subcenter = n_subcenter

#         self.set_margin(margin)
#         self.set_topK(topK, penalty)
#         self.s = scale
        
#         self.weight = nn.Parameter(torch.rand(self.n_subcenter, self.in_dim, self.n_class))
#         xavier_init(self.weight)
#         self.ce = nn.CrossEntropyLoss()
        
#     def set_margin(self, margin):
#         self.m = margin
#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)
#         self.th    = math.cos(math.pi - margin)
#         self.mm    = math.sin(math.pi - margin) * margin
        
#     def set_topK(self, topK, penalty=None):
#         self.k = topK
#         if penalty is not None:
#             self.cos_p = math.cos(penalty)
#             self.sin_p = math.sin(penalty)
        
#     def forward(self, x, label):
#         """
#         Input:
#             x: (B, D)
#             label: (B,)
#         Output:
#             output: (B, C)
#             loss: (1,)
#         """
#         # get cosine similarity from each sub-centers, then pool the maximum
#         cosine = torch.matmul(F.normalize(x, dim=1, p=2), F.normalize(self.weight, dim=1, p=2)) # (n_subcenter, B, n_class)
#         cosine, max_indice = cosine.max(dim=0) # (B, n_class)
        
#         sine   = torch.sqrt( (1.0 - torch.mul(cosine, cosine)).clamp(0, 1) )
#         phi    = cosine * self.cos_m - sine * self.sin_m
#         phi    = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
#         # one-hot label
#         one_hot = torch.zeros_like(cosine) # (B, n_class)
#         one_hot = one_hot.scatter_(-1, label[..., None], 1)
        
#         # if label; let phi be the target, else; reduce cosine similarity
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # (B, n_class)
        
#         # penalize mis-classified topK (if k==0, output=logit)
#         _, topk_indice = ((1 - one_hot) * output).topk(self.k, dim=1)
#         one_hot_k = torch.zeros_like(cosine).scatter_(-1, topk_indice, 1)

#         penalty = cosine * self.cos_p + sine * self.sin_p # cos(theta_i,j - m')
#         logit   = (one_hot_k * penalty) + ((1.0 - one_hot_k) * output)
                
#         # scale logit and calculate loss
#         logit = logit * self.s
#         loss  = self.ce(logit, label)
        
#         # get predicted class probability
#         with torch.no_grad():
#             output = output * self.s
#             output = F.softmax(output, dim=-1) # (B, n_class)
            
#         return output, loss