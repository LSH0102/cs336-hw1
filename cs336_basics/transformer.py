# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 19:36:35 2025

@author: 25488
"""

import torch
import numpy as np
from torch import nn
import einops 
from einops import einsum,rearrange
import copy

class Linear(nn.Module):
    def __init__(self,in_features, out_features,device=None,dtype=None):
        super().__init__()
        t=torch.empty(out_features,in_features)
        sigma=np.sqrt(2/(in_features+out_features))
        self.W=nn.Parameter(torch.nn.init.trunc_normal_(t,mean=0,std=sigma,a=-3*sigma,b=3*sigma))
        
    def forward(self, x:torch.Tensor):
        return torch.matmul(x, self.W.T)
    
class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        t=torch.empty(num_embeddings,embedding_dim)
        self.W=nn.Parameter(torch.nn.init.trunc_normal_(t,mean=0,std=1,a=-3,b=3))
        self.d_model=embedding_dim
    def forward(self, x:torch.LongTensor):
        seq_len=x.shape[1]
        batch_size=x.shape[0]
        y=x.flatten().unsqueeze(1)
        y=y.broadcast_to((y.shape[0],self.d_model))
        z=torch.gather(self.W, dim=0, index=y)
        
        z=z.reshape((batch_size,seq_len,self.d_model))
        return z
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g=nn.Parameter(torch.ones((d_model,)))
        self.d_model=d_model
        self.eps=eps
    def forward(self,x:torch.Tensor):
        in_type=x.dtype
        x=x.to(torch.float32)
        
        rms=torch.sqrt(torch.sum(x**2,dim=-1,keepdim=True)/self.d_model+self.eps)
        x=x/rms
        x=x*self.g
        
        return x.to(in_type)
    
class SiLU(nn.Module):
    def __init__(self,):
        super().__init__()
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x:torch.Tensor):
        return x*self.sigmoid(x)
        
class SwiGLU(nn.Module):
    def __init__(self,d_ff,d_model,device=None,dtype=None):
        super().__init__()
        self.d_ff=d_ff
        self.d_model=d_model
        
        t1=torch.empty(self.d_ff,self.d_model)
        t3=torch.empty(self.d_ff,self.d_model)
        t2=torch.empty(self.d_model,self.d_ff)
        sigma=np.sqrt(2/(self.d_ff+self.d_model))
        
        self.W1=nn.Parameter(torch.nn.init.trunc_normal_(t1,mean=0,std=sigma,a=-3*sigma,b=3*sigma))
        self.W2=nn.Parameter(torch.nn.init.trunc_normal_(t2,mean=0,std=sigma,a=-3*sigma,b=3*sigma))
        self.W3=nn.Parameter(torch.nn.init.trunc_normal_(t3,mean=0,std=sigma,a=-3*sigma,b=3*sigma))
        
        self.sigmoid=torch.nn.Sigmoid()
        
    def forward(self,x:torch.Tensor):
        a=torch.matmul(x, self.W1.T)
        a=a*self.sigmoid(a)
        
        c=torch.matmul(x, self.W3.T)
        a=a*c
        
        return torch.matmul(a, self.W2.T)
    
class RoPE(nn.Module):
    def __init__(self,theta:float, d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.d=d_k
        self.len=max_seq_len
        self.k=self.d//2 
        self.theta=theta
        
        vec=np.array([1/theta**(2*k/self.d) for k in range(0,self.k)])
        l=np.arange(self.len)
        self.grid=np.einsum('m,d->md',l,vec) #max_len x d/2
        self.sin=np.sin(self.grid)
        self.cos=np.cos(self.grid)
        
        L1=np.stack([self.cos,-self.sin],axis=-1)
        L2=np.stack([self.sin,self.cos],axis=-1)
        rotation=torch.Tensor(np.stack([L1,L2],axis=-2))
        self.rope=torch.stack([torch.block_diag(*rotation[i]) for i in range(0,self.len)],dim=0)
        
    def forward(self,x:torch.Tensor, token_positions:torch.Tensor):
        token_positions=token_positions.reshape((-1,1,1))
        y=token_positions.broadcast_to(size=(token_positions.shape[0],self.d,self.d))
        
        rot=torch.gather(self.rope, dim=0, index=y)
        
        out=einsum( x,rot,"... seq_len d_1, seq_len d_2 d_1->... seq_len d_2")
        
        return out

class Softmax(nn.Module):
    def __init__(self,dim:int):
        super().__init__()
        self.dim=dim
        
    def forward(self,x:torch.Tensor):
        max_x,_=torch.max(x,dim=self.dim,keepdim=True)
        x=x-max_x
        x=torch.exp(x)
        
        y=x/x.sum(dim=self.dim,keepdim=True)
        return y
    
def dot_atten(Q,K,V,mask=None):
    QK=einsum(Q,K.transpose(-1,-2)," batch_size ... n d_k, batch_size ... d_k m->batch_size ... n m")
    dk=K.shape[-1]
    QK=QK/np.sqrt(dk)
    if mask!=None:
        out=torch.where(mask==False, -torch.inf,0)
        QK=QK+out
        
    func=Softmax(-1)
    out=func(QK)
    out=torch.matmul(out,V)
    return out

class MultiHead_Self_Attention(nn.Module):
    def __init__(self,d_model:int ,num_heads:int, theta=None,max_seq_len=None):
        super().__init__()
        self.d_model=d_model
        self.h=num_heads
        self.dk=self.d_model//self.h
        self.dv=self.dk
        
        sigma1=2/(self.h*self.dk+self.d_model)
        sigma2=2/(self.h*self.dv+self.d_model)
        
        tq=torch.empty((self.h*self.dk,self.d_model))
        tk=torch.empty((self.h*self.dk,self.d_model))
        tv=torch.empty((self.h*self.dv,self.d_model))
        to=torch.empty((self.d_model,self.h*self.dv))
        self.wq=nn.Parameter(torch.nn.init.trunc_normal_(tq,mean=0,std=sigma1,a=-3*sigma1,b=3*sigma1))
        self.wk=nn.Parameter(torch.nn.init.trunc_normal_(tk,mean=0,std=sigma1,a=-3*sigma1,b=3*sigma1))
        self.wv=nn.Parameter(torch.nn.init.trunc_normal_(tv,mean=0,std=sigma2,a=-3*sigma2,b=3*sigma2))
        self.wo=nn.Parameter(torch.nn.init.trunc_normal_(to,mean=0,std=sigma2,a=-3*sigma2,b=3*sigma2))
        self.theta=theta
        self.rope=None
        self.max_seq_len=max_seq_len
        if self.theta!=None and self.max_seq_len!=None:
            self.rope=RoPE(self.theta, d_k=self.d_model//self.h, max_seq_len=max_seq_len)
        
    def forward(self,x:torch.Tensor,token_positions=None):  # x的最后一维应该和d_model一样
        seq_len=x.shape[-2]
        mask=torch.tril(torch.ones(size=(seq_len,seq_len))).bool()
        wq=torch.matmul(x,self.wq.transpose(-1, -2))
        wk=torch.matmul(x,self.wk.transpose(-1, -2))
        wv=torch.matmul(x,self.wv.transpose(-1, -2))
        
        wq=rearrange(wq, "... seq_len (h d_k)->... seq_len h d_k", h=self.h )
        wk=rearrange(wk, "... seq_len (h d_k)->... seq_len h d_k", h=self.h )
        wv=rearrange(wv, "... seq_len (h d_v)->... seq_len h d_v", h=self.h )
        wq=torch.split(wq, split_size_or_sections=1, dim=-2)
        wk=torch.split(wk, split_size_or_sections=1, dim=-2)
        wv=torch.split(wv, split_size_or_sections=1, dim=-2)
        if self.rope==None:
            heads=[ dot_atten(wq[i].squeeze(), wk[i].squeeze(),wv[i].squeeze(),mask=mask) for i in range(0,self.h)]
        else:
            if token_positions==None:
                token_positions=torch.LongTensor(np.arange(seq_len))
            heads=[ dot_atten(self.rope(wq[i].squeeze(),token_positions), 
                              self.rope(wk[i].squeeze(),token_positions),wv[i].squeeze(),mask=mask) for i in range(0,self.h)]
        h=torch.concat(heads,dim=-1)
        
        out=torch.matmul(h, self.wo.T)
        return out
        
class Transformer_Block(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,theta=None,max_seq_len=None):
        super().__init__()
        self.rms1=RMSNorm(d_model)
        self.rms2=RMSNorm(d_model)
        
        self.MultiHead=MultiHead_Self_Attention(d_model, num_heads,theta=theta,max_seq_len=max_seq_len)
        
        self.FFN=SwiGLU(d_ff=d_ff, d_model=d_model)
        
    def forward(self,x:torch.Tensor):
        y=x+self.MultiHead(self.rms1(x))
        x=y+self.FFN(self.rms2(y))
        return x

class Transformer(nn.Module):
    def __init__(self,vocab_size:int, context_length:int, num_layers:int, d_model:int,
                 num_heads:int,d_ff:int,theta=None,max_seq_len=None):
        super().__init__()
        
        if max_seq_len==None:
            max_seq_len=context_length
        
        self.embedding=Embedding(vocab_size, d_model)
        
        self.Trans=nn.ModuleList([copy.deepcopy(Transformer_Block(d_model, num_heads, d_ff,theta,max_seq_len)) for i in range(0,num_layers)])
        
        self.rms=RMSNorm(d_model)
        
        self.Linear=Linear(d_model, vocab_size)
        
        
        self.layers=num_layers
        
    def forward(self,x:torch.Tensor):
        x=self.embedding(x)
        
        for i in range(0,self.layers):
            x=self.Trans[i](x)
            
        x=self.rms(x)
        
        x=self.Linear(x)
        
        
        return x
    
def cross_entropy(predict,targets):
    D=predict.shape[0]
    ind,_=torch.max(predict,dim=-1,keepdim=True)
    logits=predict-ind
    x=torch.gather(logits, dim=-1, index=targets[...,None]).squeeze(-1)
    
    y=torch.log(torch.sum(torch.exp(logits),dim=-1))
    
    return -torch.sum(x-y)/D
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        