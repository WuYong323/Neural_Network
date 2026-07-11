import math

def naive_softmax(x):
    m=max(x)
    exps=[math.exp(xi-m) for xi in x]
    l=sum(exps)
    return [e/l for e in exps]

def online_softmax(x,block_size=1):
    m=float("-inf")
    l=0.0
    for i in range(0,len(x),block_size):
        block=x[i:i+block_size]
        m_block=max(block)
        m_new=max(m,m_block)
        alpha=math.exp(m-m_new)
        l=l*alpha+sum(math.exp(xi-m_new) for xi in block)
        m=m_new
        print(f"  处理到第 {i + len(block):2d} 个元素后: running_max={m:.3f}, running_sum={l:.4f}")
    return [math.exp(xi-m)/l for xi in x]


if __name__=="__main__":
    x=[2.0,5.0,1.0,7.0,3.0]
    print("=== naive(3遍,需整行) ===")
    print([round(p, 4) for p in naive_softmax(x)])
    print("=== online(1遍,分块增量,block_size=2) ===")
    p = online_softmax(x, block_size=2)
    print([round(pi, 4) for pi in p])
    # 两者输出应完全一致,证明 online softmax 数学等价、且不必先看到整行








































