import torch
from jupyterlab.semver import rtr

from rmsnorm_triton import triton_rmsnorm,_torch_rmsnorm

def test_correctness_and_shapes():
    torch.manual_seed(0)
    device="cuda"
    dim=768
    weight=torch.randn(dim,device=device,dtype=torch.bfloat16)

    for shape in [(8,1024,dim),
                  (1,1,dim),
                  (32,dim)]:
        x=torch.randn(*shape,device=device,dtype=torch.bfloat16)
        y_triton=triton_rmsnorm(x,weight)
        y_ref=_torch_rmsnorm(x,weight)
        assert torch.allclose(y_triton,y_ref,atol=1e-2,rtol=1e-2), f"形状 {shape} 不匹配！max_err={ (y_triton-y_ref).abs().max().item() }"
        print(f"{shape} 通过")


if __name__=="__main__":
    test_correctness_and_shapes()









































