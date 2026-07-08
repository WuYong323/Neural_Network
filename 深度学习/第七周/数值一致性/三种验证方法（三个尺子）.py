import torch
import torch.nn.functional as F

@torch.no_grad()
def error_report(logits_ref,logits_test,name="test"):
    ref=logits_ref.float().flatten(end_dim=-2)
    test=logits_test.float().flatten(end_dim=-2)

    # 尺子①: 逐元素接近(量化下通常 False, 只看误差量级)
    max_abs=(ref-test).abs().max().item()
    allclose=torch.allclose(ref,test,rtol=1e-2,atol=1e-2)

    # 尺子②: logits 余弦相似度(量化的主度量, 看决策方向)
    cos=F.cosine_similarity(ref,test,dim=-1).mean().item()

    # 尺子③(近似): top-1 一致率——argmax 选的 token 有多少比例没变
    top1_agree=(ref.argmax(-1)==test.argmax(-1)).float().mean().item()

    print(f"[{name}] 最大逐元素误差 : {max_abs:.4e}")
    print(f"[{name}] allclose       : {allclose}  (量化下 False 正常)")
    print(f"[{name}] logits cosine  : {cos:.6f}  (>0.99 才算方向一致)")
    print(f"[{name}] top-1 一致率   : {top1_agree:.4f}  (=1.0 表示每步选的 token 都没变)")
    return cos, top1_agree

torch.manual_seed(0)
linear=torch.nn.Linear(512,5000)
x=torch.randn(8,512)

logits_fp32=linear(x)
logits_fp16=linear.half()(x.half()).float()
error_report(logits_fp32,logits_fp16,"FP16")










































