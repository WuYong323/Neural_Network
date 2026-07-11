import numpy as np
from sympy.abc import alpha


def flash_attention_single_query(q,k,v):
    """
    q: (d,)      单个 query 向量
    K: (T, d)    所有 key
    V: (T, d)    所有 value
    返回: (d,)   该 query 的注意力输出,等价于 softmax(qKᵀ/√d) @ V
    """
    d=q.shape[0]
    scale=1.0/np.sqrt(d)
    m=-np.inf
    l=0.0
    acc=np.zeros(d)
    for j in range(k.shape[0]):
        s=np.dot(q,k[j])*scale     # 当前这一个分数,一个标量
        m_new=max(m,s)
        alpha=np.exp(m-m_new)
        p=np.exp(s-m_new)
        l=l*alpha+p
        acc=acc*alpha+p*v[j]
        m=m_new
    return acc/l

if __name__=="__main__":
    rng=np.random.default_rng(0)
    T,d=64,32
    q=rng.standard_normal(d)
    k=rng.standard_normal((T,d))
    v=rng.standard_normal((T,d))

    out_flash=flash_attention_single_query(q,k,v)

    # 参考:标准三步法(显式造出长度 T 的权重向量)
    scale=1.0/np.sqrt(d)
    s=(q@k.T)*scale
    p=np.exp(s-s.max())
    p/=p.sum()
    out_ref=p@v

    cos = np.dot(out_flash, out_ref) / (np.linalg.norm(out_flash) * np.linalg.norm(out_ref))
    print(f"cosine 相似度 = {cos:.8f}  (应 ≈ 1.0)")
    # 注意 acc 始终是 (d,) 维,循环里从没出现过 (T,) 的完整权重向量









































