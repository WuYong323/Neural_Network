import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import num_warps


def _triton_available():
    try:
        import triton
        return True
    except ImportError:
        return False


@triton.jit
def _rmsnorm_fwd_kernel(
        x_ptr,w_ptr,y_ptr,
        x_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE:tl.constexpr,
):
    row_idx=tl.program_id(0)
    x_ptr_row=x_ptr+row_idx*x_row_stride

    col_offsets=tl.arange(0,BLOCK_SIZE)
    mask=col_offsets<n_cols

    x=tl.load(x_ptr_row+col_offsets,mask=mask,other=0.0)

    x_fp32=x.to(tl.float32)
    mean_sq=tl.sum(x_fp32*x_fp32,axis=0)/n_cols
    rrms=1.0/tl.sqrt(mean_sq)

    w=tl.load(w_ptr+col_offsets,mask=mask,other=0.0)

    y=x_fp32*rrms*w.to(tl.float32)

    y_ptr_row=y_ptr+row_idx*x_row_stride
    tl.store(y_ptr_row+col_offsets,y,mask=mask)



def triton_rmsnorm(x:torch.Tensor,weight:torch.Tensor,eps:float=1e-6):
    """
    对最后一维做 RMSNorm。x 可以是任意前置维度，如 [B,S,D] 或 [B,D]。
    """
    orig_shape=x.shape
    dim=orig_shape[-1]
    x2d=x.reshape(-1,dim)
    x2d=x2d.contiguous()
    n_rows,n_cols=x2d.shape

    y=torch.empty_like(x2d)

    BLOCK_SIZE=triton.next_power_of_2(n_cols)
    grid=(n_rows,)

    _rmsnorm_fwd_kernel[grid](
        x2d,weight,y,
        x2d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE<=2048 else 8,
    )
    return y.reshape(orig_shape)


def _torch_rmsnorm(x,weight,eps=1e-6):
    dtype=x.dtype
    x=x.to(torch.float32)

    x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)
    return (x*weight).to(dtype)






































