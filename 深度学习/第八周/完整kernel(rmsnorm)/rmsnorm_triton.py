import torch
import triton
import triton.language as tl

def _triton_available():
    try:
        import triton  # noqa
        return True
    except ImportError:
        return False

@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,          # 输入张量首地址（已 reshape 成 [n_rows, n_cols]）
    w_ptr,          # weight 首地址，长度 n_cols
    y_ptr,          # 输出张量首地址
    x_row_stride,   # 相邻两行在内存里差多少个元素（通常 = n_cols）
    n_cols,         # 每行元素个数（= 模型 hidden dim）
    eps,            # 防除零小量
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：一次处理的列数（取 >= n_cols 的 2 的幂）
):
    # 每个 program 负责一行。program_id(0) 就是行号。
    # 为什么按行并行：RMSNorm 逐行独立，行与行之间没有数据依赖，天然可并行。
    row_idx = tl.program_id(0)
    x_ptr_row = x_ptr + row_idx * x_row_stride  # 定位到本行起始地址

    # 列方向的偏移 [0,1,...,BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 掩码：BLOCK_SIZE 可能比 n_cols 大，超出部分不能读，用 mask 挡住
    mask = col_offsets < n_cols

    # 读入本行。other=0.0：被 mask 挡住的位置填 0，不影响后面平方和。
    x = tl.load(x_ptr_row + col_offsets, mask=mask, other=0.0)

    # 关键：转 fp32 再算平方和。为什么？见 §2.1——归一化的 reduction 必须高精度，
    # 否则 bf16/fp16 累加会掉精度甚至溢出。这是工业级实现的铁律。
    x_fp32 = x.to(tl.float32)
    mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols  # mean(x²)
    rrms = 1.0 / tl.sqrt(mean_sq + eps)                 # 1/rms，用乘代替除更快

    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    # 先在 fp32 下缩放，乘 weight，最后 tl.store 时自动转回 y 的 dtype
    y = x_fp32 * rrms * w.to(tl.float32)

    y_ptr_row = y_ptr + row_idx * x_row_stride
    tl.store(y_ptr_row + col_offsets, y, mask=mask)     # 写回，mask 保证不越界

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    对最后一维做 RMSNorm。x 可以是任意前置维度，如 [B,S,D] 或 [B,D]。
    """
    orig_shape = x.shape
    dim = orig_shape[-1]
    # 把所有前置维度压平成"行"。为什么：kernel 只认 [n_rows, n_cols]，
    # 这样无论 [B,S,D] 还是 [B,1,D]（decode）都能统一处理——解决 §2.1 的形状问题。
    x2d = x.reshape(-1, dim)
    x2d = x2d.contiguous()          # 确保内存连续，否则 stride 假设不成立会读错
    n_rows, n_cols = x2d.shape

    y = torch.empty_like(x2d)
    # BLOCK_SIZE 取 >= n_cols 的最小 2 的幂：让一行一次性读完，reduction 最简单
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)                # 启动 n_rows 个 program，一行一个

    _rmsnorm_fwd_kernel[grid](
        x2d, weight, y,
        x2d.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,  # 列多时多给 warp，提升并行度
    )
    return y.reshape(orig_shape)    # 还原成原始形状，对调用方透明

def _torch_rmsnorm(x, weight, eps=1e-6):
    """纯 PyTorch 参考实现，用作 fallback 和 §7 的正确性基准。"""
    dtype = x.dtype
    x = x.to(torch.float32)                                    # 同样 fp32 累加
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(dtype)                              # 最后转回原 dtype