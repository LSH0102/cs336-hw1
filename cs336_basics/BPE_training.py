# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:25:27 2025

@author: 25488
"""

#find_chunk_boundaries 复制自 pretokenization_example.py
import os
from typing import BinaryIO
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




import regex as re
import numpy as np


def compute_chunks(input_path:str,special_tokens:None):
    d_list=[]
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            d=pre_tokenization(chunk,special_tokens )
            d_list.append(d)
    return d_list
            
def pre_tokenization(text:str,special_tokens):
    
    d=dict()
    if special_tokens!=None:
        sp=[re.escape(token) for token in special_tokens]
        delimiter = "|".join(sp) 
            
        docs=re.split(f"{delimiter}",text)
    else:
        docs=[text]
    for doc in docs:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if doc in special_tokens:
            continue
        data=re.finditer(PAT,doc)
        for match in data:
            token=match.group(0)
            code=tuple([bytes([b]) for b in token.encode("utf-8")])          #最好在这里拆分成bytes 否则后面会报错
            if code in d:
                d[code]+=1
            else:
                d[code]=1
    
    return d

def construct_all_pre_tokens(input_path:str,special_tokens:None):
    L=compute_chunks(input_path,special_tokens)
    d=dict()
    for dictionary in L:
        for k,v in dictionary.items():
            if k in d:
                d[k]+=v
            else:
                d[k]=v
    return d

def merge(current_vocab:dict, merges:list, alphabet: dict):
    
    #首先构建相邻字符的组合:      另外 在这个过程中可以通过动态维护频率最高的组合来获得merge
    all_combinations=dict()
    maximum=0
    target=None
    for k,v in alphabet.items():                #遍历整个字典 
        for j in range(0,len(k)-1):
            s=tuple([k[j],k[j+1]])
            if s in all_combinations:
                all_combinations[s]+=v
            else:
                all_combinations[s]=v
            #如果s的频率是迄今为止最大的 那么更新最大频率和target列表
            if all_combinations[s]>maximum:
                target=s
                maximum=all_combinations[s]
            #如果s的频率恰好等于最大频率 那么把s放入target列表 一会抽出字典序最大的那个
            elif all_combinations[s]==maximum:
                if s>target:
                    target=s
            #如果小于 那么什么都不做
    
    
    #此时target已经成为了我们要更新的tuple, 把target扔进vocab和merge表里面:
    current_vocab[len(current_vocab)]=target[0]+target[1]
    merges.append((target[0],target[1]))
    
    #作为最后一步, 需要更新alphabet
    d=dict()
    for k,v in alphabet.items():     
        List=[]
        j=0
        while j<len(k):
            if j!=len(k)-1:             #因为每一组merge都不可能是token的最后一个字符, 所有            
                if k[j]==target[0] and k[j+1]==target[1]:       #如果找到了对应的组合
                    List.append(target[0]+target[1])
                    j=j+1   #找到之后注意跳过下一个j, 要不然会有重复
                    
                else:
                    List.append(k[j])       #如果不是则什么都不改变
            else:
                List.append(k[j])
            j=j+1
                
        
        d[tuple(List)]=v
                
    #重新更新alphabet     
    alphabet=d
    return current_vocab,merges,alphabet
    


def BPE_train(input_path:str,vocab_size:int,special_tokens:list[str]):
    #首先创建初始化的vocab, 由256个可能的字节值组成
    current_vocab=dict()
    for i in range(0,256):
        current_vocab[i]=bytes([i])
    for i in range(0,len(special_tokens)):
        current_vocab[256+i]=special_tokens[i].encode("utf-8")
    
    pre_tokens=construct_all_pre_tokens(input_path,special_tokens)
    #把pre_token拆成单个字符dict[tuple(bytes),int] 不能存成list(bytes) 因为不可哈希
    alphabet=dict()
    for k,v in pre_tokens.items():
        alphabet[tuple(list(k))]=v
    
    merges=[]
    #开始merge, 每一次merge都会带来一个新的token, 不断扩大vocab字典d, 直到达到vocab_size
    for i in range(len(current_vocab),vocab_size):
        current_vocab,merges,alphabet=merge(current_vocab=current_vocab,merges=merges,alphabet=alphabet)
    return current_vocab,merges

if __name__=="__main__":
    #测试merge是否正确
    pre_tokens={'low':5, 'lower': 2, 'widest':3, 'newest':6}
    alphabet=dict()
    for k,v in pre_tokens.items():
        alphabet[tuple(list(k))]=v
    current_vocab=dict()
    merges=[]
    
    for i in range(0,6):
        current_vocab,merges,alphabet=merge(current_vocab=current_vocab,merges=merges,alphabet=alphabet)
        
    print(current_vocab)
    print(merges)
    print(alphabet)
    
    
    























    