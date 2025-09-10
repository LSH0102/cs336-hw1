# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 08:46:55 2025

@author: 25488
"""

import torch
import numpy as np
import os
import typing

def get_data(x:np.ndarray,batch_size:int, context_length:int,device):
    inds=np.random.randint(low=0,high=x.shape[0]-context_length,size=(batch_size,))   #保证一定的随机性
    batch=[x[i:i+context_length] for i in inds]
    label=[x[i+1:i+context_length+1] for i in inds]
    batch=np.array(batch)
    label=np.array(label)
    batch=torch.LongTensor(batch,device=device)
    label=torch.LongTensor(label,device=device)
    return batch,label

def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer,
                    iteration:int,out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    check_point={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"iteration":iteration}
    torch.save(check_point,out)
    
    
def load_checkpoint(src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model:torch.nn.Module,optimizer:torch.optim.Optimizer):
    check_point=torch.load(src)
    model.load_state_dict(check_point["model"])
    optimizer.load_state_dict(check_point["optimizer"])
    return check_point["iteration"]
    