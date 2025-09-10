# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:43:47 2025

@author: 25488
"""

from collections.abc import Iterable
from typing import Iterator
import json
import regex as re

class Tokenizer:
    def __init__(self,vocab, merges, special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens          #排序防止出现子串问题
        if self.special_tokens!=None:
            self.special_tokens=sorted(special_tokens,key=lambda x:-len(x))
        self.token2id=dict()
        for k,v in self.vocab.items():
            self.token2id[v]=k
            
    def compute_merge(self,word):
        target=word
        for k in range(0,len(word)+3):             #至多循环这么多次就可以跳出
            x=[]
            j=0
            possible_merges=[]
            possible_index=[]
            target_index=[]
            for j in range(0,len(target)):
                if j<len(target)-1:              #可能得merge只会在前n-1个token出现
                    if (target[j],target[j+1]) in self.merges:
                        possible_merges.append((target[j],target[j+1]))
                        possible_index.append(self.merges.index((target[j],target[j+1])))
                        target_index.append(j)
            #找到所有可能得merge之后 如果没有则结束循环, 查看下一个word, 否则抽出字典序最大的那个
            if len(possible_index)==0:
                
                return target
            merge_index=possible_index.index(min(possible_index))     
            merge=possible_merges[merge_index]
            ind=target_index[merge_index]             #找到这个merge左边在target中的下标 保证merge前后元素不变
            #修改target
            x=[('','')]*(len(target)-1)
            x[:ind]=target[:ind]
            x[ind]=merge[0]+merge[1]
            x[ind+1:]=target[ind+2:]
            target=x
        
        
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,'r',encoding='utf-8') as f:
            self.vocab=json.load(f)
        with open(merges_filepath,'r',encoding='utf-8') as f:
            self.merges=json.load(f)
        if special_tokens==None:
            self.special_tokens=None
        else:
            with open(special_tokens,'r',encoding='utf-8') as f:
                self.special_tokens=json.load(f)
        
    def encode(self, text: str) -> list[int]:
        #边界情形直接输出
        if len(text)==0:
            return []
        #先做pre-tokenization
        sequence=[]
        if self.special_tokens!=None:
            sp=[f"({re.escape(token)})" for token in self.special_tokens]
            sp = "|".join(sp) 
                
            docs=re.split(sp,text)
            
            docs=[doc for doc in docs if doc!=None]
        else:
            docs=[text]
        special=[]
        count=[]
        t=0
        for doc in docs:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            if self.special_tokens!=None:
                if doc in self.special_tokens:
                    sequence.append([doc.encode("utf-8")])
                    count.append(t)
                    t+=1
                    continue
            data=re.finditer(PAT,doc)
            L=[]
            for match in data:                     #这次不要拆分成bytes, 直接words
                token=match.group(0)
                
                L.append(list([bytes([b]) for b in token.encode("utf-8")]))
            sequence.append(L)
            t+=1
        token_seq=[]
        t=0
        for seq in sequence:
            #首先判断这个word是不是special_token
            if t in count:
                token_seq.append([seq[0]])    #如果是特殊字符那么seq长度为1
            else:
                for word in seq:
                    token_seq.append(self.compute_merge(word))
            t+=1
        id_seq=[]
        
        for word in token_seq:
            for token in word:
                id_seq.append(self.token2id[token])
        return id_seq
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        
        for text in iterable:
            for i in self.encode(text):
                yield i
    def decode(self, ids: list[int]) -> str:
        #边界情形直接输出
        if len(ids)==0:
            return ''
        else:
            b=[self.vocab[i] for i in ids]
            b=b''.join(b)
            s=b.decode("utf-8",errors='replace')
        
        return s
    
    
    