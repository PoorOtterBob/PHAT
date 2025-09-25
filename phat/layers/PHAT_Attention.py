import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from ts_benchmark.baselines.iclr.layers.DyT import DyT

class PHAT_Attention(nn.Module):
    def __init__(self, model_dim, head, dropout=0, depth=1, 
    is_poa=True, is_paa=True, is_poa_diff=True, is_poa_mod=True,
    is_lambda_res=True, is_group_norm=True,
    ):
        super(PHAT_Attention, self).__init__()
        self.model_dim = model_dim
        self.head_num = head
        self.head_dim = self.model_dim // self.head_num
        self.key_scaling = self.head_dim ** (-0.5)
        self.attn_dropout = dropout
        assert model_dim % head == 0, "dim must be divisible by head"
        assert (self.head_num % 2 == 0), "differential head need be devided by 2!"

        self.is_poa = is_poa
        self.is_poa_diff = is_poa_diff
        self.is_poa_mod = is_poa_mod
        self.is_paa = is_paa
        self.is_lambda_res = is_lambda_res
        self.is_group_norm = is_group_norm

        self.transformation = nn.Sequential(
                                nn.Linear(self.model_dim, 3*self.model_dim),
                                nn.Unflatten(dim=-1, unflattened_size=(self.head_num//2, 3*self.head_dim, 2)),
                            )
        
        self._lambda = nn.Sequential(
            nn.Linear(self.model_dim, self.head_num//2),
            nn.Sigmoid(),
        )
        
        if  self.is_group_norm:
            self.output = nn.Sequential(
                DyT(self.head_dim*2),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(self.head_num * self.head_dim, self.model_dim),
                )
        else:
            self.output = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(self.head_num * self.head_dim, self.model_dim),
                )
        self._init_param()
        
    def forward(self, x):
        attn_offset = None
        attn_align = None
        
        _lambda = self._lambda(x)
        # (B, P, N, D)
        q, k, v = torch.chunk(self.transformation(x), chunks=3, dim=-2) # (B, P, N, D) -> (B, P, N, H, HD, 2)
        k = k*self.key_scaling
        v = torch.flatten(v, start_dim=-2, end_dim=-1) # (B, P, N, H, HD, 2) -> (B, P, N, H, 2*HD)

        logits_offset = torch.einsum('binhde, bjnhde -> bijnhe', 
                                     q, 
                                     k)
        # Offset Attention
        if self.is_poa:
            if self.is_poa_mod:
                forget_ind_offset = self._forget_bias(x.shape[1], x.shape[2]>1, x.shape[2]>1, x.device)
                forget_offset = torch.einsum('biknhe, ikje -> bijnhe', 
                                            self._softplus(logits_offset), 
                                            forget_ind_offset)
                attn_offset = torch.softmax(logits_offset - forget_offset, dim=2) # (B, P, P, N, H, HD, 2)
            else:
                attn_offset = torch.softmax(logits_offset, dim=2) # (B, P, P, N, H, HD, 2)
            if self.is_poa_diff:
                attn_offset = attn_offset[..., 0] - _lambda[:, None]*attn_offset[..., -1]
            else:
                attn_offset = attn_offset[..., 0]

        # Align Attention
        if x.shape[2] > 1 and self.is_paa:
            attn_align = torch.softmax(torch.einsum('bpihd, bpjhd -> bpijh', 
                                                    q[..., 0],
                                                    k[..., 0]),
                                        dim=-2)

        attn = self._attn(v, attn_offset, attn_align)

        
        if self.is_lambda_res:
            attn = attn + _lambda[..., None] * torch.unflatten(x, -1, (self.head_num//2, -1))
        attn = self.output(attn)
        
        return attn

    def _attn(self, value, attn_offset=None, attn_align=None):   
        if attn_align is not None:
            value = torch.einsum('bpnjh, bpjhd -> bpnhd', attn_align, value)
        if attn_offset is not None:
            value = torch.einsum('bpinh, binhd -> bpnhd', attn_offset, value)
        return value

    def _init_param(self):
        self.transformation.apply(lambda x: nn.init.xavier_uniform_(x.weight) if isinstance(x, nn.Linear) else None)
        self.transformation.apply(lambda x: nn.init.constant_(x.bias, 0) if isinstance(x, nn.Linear) else None)
        self.output.apply(lambda x: nn.init.xavier_uniform_(x.weight) if isinstance(x, nn.Linear) else None)
        self.output.apply(lambda x: nn.init.constant_(x.bias, 0) if isinstance(x, nn.Linear) else None)


    def _forget_bias(self, period_length, is_cyclic, is_diff, device):
        pos = torch.arange(period_length, device=device, requires_grad=False)
        rel_pos = torch.abs(pos[:, None] - pos[None, :])
        
        if is_cyclic:
            rel_pos = torch.min(rel_pos % period_length, (-rel_pos) % period_length)

        i = pos.view(-1, 1, 1)
        k = pos.view(1, -1, 1)
        j = pos.view(1, 1, -1)

        mask_k_ne_i = (k != i).float()
        mask_j_eq_k = (j == k).float()

        forget_index = (rel_pos[i, k] < rel_pos[i, j]).float() * mask_k_ne_i + mask_j_eq_k

        if is_diff:
            forget_diff_index = (rel_pos[i, k] > rel_pos[i, j]).float() * mask_k_ne_i + mask_j_eq_k
        else:
            forget_diff_index = forget_index     
        forget_index = torch.stack([forget_index, forget_diff_index], dim=-1)

        return forget_index
 
    def _softplus(self, x):
        return torch.where(x < math.log(32768), F.softplus(x), x)
