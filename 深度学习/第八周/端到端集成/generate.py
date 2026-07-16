# generate.py —— 最小可用的自回归生成，推理态只读、不建计算图
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model,idx,max_new_tokens,temperature=1.0,top_k=None):
    """
    idx: [B, T] 初始 token（prompt）
    这是不带 KV Cache 的朴素版：每步把全序列重算一遍。
    真实高性能推理会用 KV Cache（§2.3）只算最新 token——这里为聚焦"kernel 集成"先简化。
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond=idx if idx.size(1) <=model.config["block_size"] else idx[:,-model.config["block_size"]:]
        logits,_=model(idx_cond)
        logits=logits[:,-1,:]/temperature
        if top_k is not None:
            v,_=torch.topk(logits,min(top_k,logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs=F.softmax(logits,dim=-1)
        idx_nest=torch.multinomial(probs,num_samples=1)
        idx=torch.cat((idx,idx_nest),dim=1)
    return idx








































