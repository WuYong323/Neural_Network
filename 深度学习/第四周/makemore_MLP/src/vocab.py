from typing import Iterable

def build_vocab(words:Iterable[str]) -> tuple[dict[str,int],dict[int,str]]:
    chars=sorted(set("".join(words)))
    stoi={".":0}
    stoi.update({ch:i+1 for i,ch in enumerate(chars)})
    itos={i:ch for ch,i in stoi.items()}
    return stoi,itos

vocab_size=27

