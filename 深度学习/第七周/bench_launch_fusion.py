import os

from torch._C import dtype

os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import torch


assert torch.cuda.is_available(),"本脚本需要CUDA GPU"
device="cuda"
torch.manual_seed(0)
print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")



def cuda_time_ms(fn,iters=200,warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters

# ---------- 实验 1: 纯 launch overhead ----------
x_small=torch.randn(16,device=device)
N=50


def many_small():
    y=x_small
    for _ in range(N):
        y=y+1.0
    return y

def one_op():
    return x_small+1.0

t_many=cuda_time_ms(many_small)
t_one=cuda_time_ms(one_op)
per_launch_us=(t_many-t_one)/(N-1)*1000.0   # ms->us, 摊到每次增量 launch
print("\n===== 实验1: 单次 kernel launch 固定开销 =====")
print(f"  {N} 个小add串起来：  {t_many*1000:8.2f} us")
print(f"  1 个小add（基线）：    {t_one*1000:8.2f} us")
print(f" =>单次 launch 增量开销： {per_launch_us:6.2f} us/次    (与算多少无关)")


# ---------- 实验 2: 融合的第二笔账"省访存" ----------
n=16000000
a=torch.randn(n,device=device,dtype=torch.float32)
b=torch.randn(n,device=device,dtype=torch.float32)
c=torch.randn(n,device=device,dtype=torch.float32)


def unfuse():
    tmp=a*b
    return tmp+c

def fused():
    return torch.addcmul(c,a,b)


t_unfused=cuda_time_ms(unfuse)
t_fused=cuda_time_ms(fused)
print("\n===== 实验2: 融合省访存(out = a*b + c) =====")
print(f"  未融合(mul+add):      {t_unfused*1000:8.2f} us   (中间量 tmp 往返 HBM + 2 次 launch)")
print(f"  融合(addcmul):        {t_fused*1000:8.2f} us   (中间量留寄存器 + 1 次 launch)")
print(f"  => 加速: {t_unfused/t_fused:5.2f}x")


# ---------- 实验 3: CUDA Graph 把 N 次 launch 压成 1 次 ----------
def replay_workload():
    global x_small
    y=x_small
    for _ in range(N):
        y=y+1.0
    return y

# 录制前必须先热身几步(捕获期不允许有 cudaMalloc 等一次性初始化)
s=torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        replay_workload()
torch.cuda.current_stream().wait_stream(s)

g=torch.cuda.CUDAGraph()
static_out=None
with torch.cuda.graph(g):
    y=x_small
    for _ in range(N):
        y=y+1.0
    static_out=y

t_eager_launch=cuda_time_ms(replay_workload)
t_graph=cuda_time_ms(g.replay)
print("\n===== 实验3: CUDA Graph 绕过 CPU launch 开销 =====")
print(f"  eager 逐个 launch({N}次): {t_eager_launch*1000:8.2f} us")
print(f"  CUDA Graph replay(1次):  {t_graph*1000:8.2f} us")
print(f"  => 加速: {t_eager_launch/t_graph:5.2f}x   (省的就是 CPU 发射开销)")





























