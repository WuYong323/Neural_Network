import torch
from torch.profiler import profile,ProfilerActivity,record_function
from KV_cache import miniGPT,generate_kv



device="cuda" if torch.cuda.is_available() else "cpu"
model=miniGPT(vocab_size=65,n_layer=4,n_head=4,n_embd=128,block_size=512).to(device).eval()
idx=torch.randint(0,65,(1,8),device=device)

activities=[ProfilerActivity.CPU]
if device=="cuda":
    activities.append(ProfilerActivity.CUDA)

with torch.no_grad():
    generate_kv(model,idx.clone(),32)
if device=="cuda":
    torch.cuda.synchronize()

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True
)as prof:
    with torch.no_grad():
        with record_function("PREFILL+DECODE_generate_256"):
            generate_kv(model,idx.clone(),256)
        if device=="cuda":
            torch.cuda.synchronize()

prof.export_chrome_trace("logs/nanogpt_trace.json")
print("trace 已导出到 logs/nanogpt_trace.json")

sort_key="cuda_time_total" if device=="cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key,row_limit=12))








































