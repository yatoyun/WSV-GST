import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

class DistanceAdj(nn.Module):
    def __init__(self, sigma, bias):
        super(DistanceAdj, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).cuda()
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1)

        return dist


class TCA(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads, norm=None):
        super(TCA, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k  # same as dim_q
        self.n_heads = n_heads
        self.norm = norm

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)

        self.norm_fact = 1 / math.sqrt(dim_k)
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, mask, adj=None):
        Q = self.q(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(x).view(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)

        if adj is not None:
            g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact + adj
        else:
            g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        l_map = g_map.clone()
        l_map = l_map.masked_fill_(mask.data.eq(0), -1e9)

        g_map = self.act(g_map)
        l_map = self.act(l_map)
        glb = torch.matmul(g_map, V).view(x.shape[0], x.shape[1], -1)
        lcl = torch.matmul(l_map, V).view(x.shape[0], x.shape[1], -1)

        alpha = torch.sigmoid(self.alpha)
        tmp = alpha * glb + (1 - alpha) * lcl
        if self.norm:
            tmp = torch.sqrt(F.relu(tmp)) - torch.sqrt(F.relu(-tmp))  # power norm
            tmp = F.normalize(tmp)  # l2 norm
        tmp = self.o(tmp).view(-1, x.shape[1], x.shape[2])
        return tmp


class Pdropout(nn.Module):
    """"
    reference:
    https://github.com/ChongQingNoSubway/PDL
    """
    def __init__(self,p=0):
        super(Pdropout,self).__init__()
        if not(0 <= p <= 1):
            raise ValueError("Drop rate must be in range [0,1]")
        self.p = p 

    def forward(self,input_data):
        if not self.training:
            return input_data
        else:
            b, n, f = input_data.shape
            input_data = input_data.view(-1,f)
            importances = torch.mean(input_data,dim=1,keepdim=True)
            importances = torch.sigmoid(importances)
            mask = self.generate_mask(importances,input_data)
            
            input_data = input_data*mask
            
            input_data = input_data.view(b,n,f)
            return input_data
        
    def generate_mask(self,importance,input_data):
        n,f = input_data.shape
        interpolation = self.non_linear_interpolation(self.p,0,n).to(input_data.device)
        mask = torch.zeros_like(importance)
        mask = mask.to(input_data.device)
        _, indx = torch.sort(importance,dim=0)
        idx = indx.view(-1)
        # Ensure the shapes match before the index_add_ operation
        interpolation = interpolation.view(-1, 1)  # Adjust shape to match mask's
        mask.index_add_(0, idx, interpolation)
        
        #mask 
        sampler = torch.rand(mask.shape[0],mask.shape[1]).to(input_data.device)
        mask = (sampler < mask).float()
        mask = 1 - mask
        return mask
    
    def non_linear_interpolation(self,max,min,num):
        e_base = 20
        log_e = 1.5
        res = (max - min)/log_e* np.log10((np.linspace(0, np.power(10,(log_e)) - 1, num)+ 1)) + min
        res = torch.from_numpy(res).float()
        return res
    
class InstanceLevelDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(InstanceLevelDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        """
        Forward pass for instance-level dropout.

        Parameters:
            x (Tensor): input_data tensor of shape (batch_size, frames, features).

        Returns:
            Tensor: Output tensor after instance-level dropout.
        """
        if self.training:
            # Create a binary mask with shape (batch_size, frames)
            mask = torch.rand(x.size(0), x.size(1)) < (1 - self.dropout_prob)

            # Move mask to the same device as the input_data tensor
            mask = mask.to(x.device).float()

            # Expand mask to have the same shape as x
            mask = mask.unsqueeze(-1).expand_as(x)

            # Apply mask
            x = x * mask

        return x
