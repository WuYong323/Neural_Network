from __future__ import annotations
import torch
from torch import Tensor

def build_dataset(words:list[str],stoi:dict[str,int],block_size:int =3)-> tuple[Tensor,Tensor]:
    assert "." in stoi, "vocab must contain '.' as start/end token"

    #类型注解
    X_rows:list[list[int]]=[]
    Y_rows:list[int]=[]
    pad_id=stoi["."]

    for word in words:
        context=[pad_id]*block_size
        for ch in word+".":
            target=stoi[ch]
            X_rows.append(context.copy())
            Y_rows.append(target)
            context=context[1:]+[target]

    X=torch.tensor(X_rows,dtype=torch.long)
    Y=torch.tensor(Y_rows,dtype=torch.long)

    return X,Y

