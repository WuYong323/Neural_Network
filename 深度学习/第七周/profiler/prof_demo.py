import torch
from torch.profiler import profile,ProfilerActivity,schedule,record_function
# 1.一个性能分析器（上下文管理器）   2.决定要记录什么硬件的活动（枚举类）   3. 制定分析的节拍与周期（辅助函数）  4. 代码区块贴上自定义标签


device="cuda" if torch.cuda.is_available() else "cpu"
# 模拟decode：一个token（M=1）过一层MLP。注意M=1正是decode的 GEMV 形态
hidden,inter=4096,11008
x=torch.randn(1,hidden,device=device,dtype=torch.float16)
w1=torch.randn(hidden,inter,device=device,dtype=torch.float16)
w2=torch.randn(inter,hidden,device=device,dtype=torch.float16)

def decode_step():
    with record_function("mlp_block"):
        h=torch.relu(x@w1)
        out=h@w2
    return out

my_schedule=schedule(wait=1,warmup=2,active=3,repeat=1)

with profile(
    activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
    schedule=my_schedule,
    record_shapes=True,   # 记录每个 op 的输入张量形状(排查"为什么这个 mm 这么慢"靠它)
    profile_memory=True,  # 记录显存分配/释放(排查 OOM、看 cat 这类算子的显存行为)
    with_stack=False,     # 记录 Python 调用栈(精确但更慢,定位"哪行代码"时才开)
) as prof:
    for _ in range(6):    # 步数要 >= wait+warmup+active = 6
        decode_step()
        prof.step()       # 关键: 告诉 profiler "一步结束了",它据此推进 schedule

#   "self" = 只算这个 op 自己的时间, 不含它调用的子 op, 避免父子重复计数
print(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=10))
# key_average   把原始记录按算子的“键”进行聚合统计
# table   将聚合后的统计结果格式化为一个易于阅读的 ASCII 表格字符串

prof.export_chrome_trace("trace.json")






































