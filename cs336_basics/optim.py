# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 09:19:11 2025

@author: 25488
"""

import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from collections.abc import Callable,Iterable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8,):
        
        defaults={"lr":lr,"weight_decay":weight_decay,"betas":betas,"eps":eps}
        super().__init__(params,defaults)
        
        
    def step(self,closure:Optional[Callable]=None):
        loss=None if closure is None else closure()
        
        for group in self.param_groups:
            lr=group["lr"]
            beta1=group['betas'][0]
            beta2=group['betas'][1]
            alpha=group['lr']
            eps=group['eps']
            weight_decay=group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                t=state.get("t",1)             #获取迭代次数
                grad=p.grad.data
                if 'm' not in state:
                    state['m']=torch.zeros_like(p.grad)
                    state['v']=torch.zeros_like(p.grad)
                state['m']=beta1*state['m']+(1.0-beta1)*grad
                state['v']=beta2*state['v']+(1.0-beta2)*(grad**2)
                alpha_t=alpha*math.sqrt(1.0-beta2**t)/(1.0-beta1**t)
                p.data-=alpha_t*state['m']/(torch.sqrt(state['v'])+eps)
                p.data-=alpha*weight_decay*p.data
                state['t']=t+1
        return loss
    
def scheduler(t,a_max,a_min,Tw,Tc):
    if t<Tw:
        return t/Tw*a_max
    elif t<=Tc:
        return a_min+0.5*(1.0+math.cos((t-Tw)/(Tc-Tw)*math.pi))*(a_max-a_min)
    else:
        return a_min
    
def grad_clipping(params:list,M):
    '梯度裁剪是裁剪所有元素得梯度'
    norm=0.0
    for p in params:
        if p.grad is None:
            continue
        else:
            norm+=torch.norm(p.grad,p=2)**2
    norm=math.sqrt(norm)
    if norm>=M:
        for p in params:
            if p.grad is None:
                continue
            else:
                p.grad.data*=M/(norm+1e-6)
                
                
                
        
        
    