import torch,struct

def bits_of(x_tensor):
    if x_tensor.dtype==torch.float32:
        b=x_tensor.view(torch.int32).item()&0xFFFFFFFF
        w=32
    elif x_tensor.dtype==torch.float16:
        b=x_tensor.view(torch.int16).item()&0xFFFF
        w=16
    elif x_tensor.dtype==torch.bfloat16:
        b=x_tensor.view(torch.int16).item()&0xFFFF
        w=16
    return format(b,f'0{w}b')

val=1.5
f32=torch.tensor(val,dtype=torch.float32)
f16=f32.half()          #转 FP16
bf16=f32.bfloat16()     #转 BF16

s32=bits_of(f32)
print(f"FP32 : {s32[0]} {s32[1:9]} {s32[9:]}")     # 1符号 8指数 23尾数
sb = bits_of(bf16)
print(f"BF16 : {sb[0]} {sb[1:9]} {sb[9:]}")        # 1符号 8指数 7尾数
# ★ 关键验证: BF16 的 16 位, 恰好等于 FP32 的高 16 位!
print(f"FP32 高16位 == BF16 ? {s32[:16] == sb}")   # 预期 True (1.5 这类整洁的数)
sh = bits_of(f16)
print(f"FP16 : {sh[0]} {sh[1:6]} {sh[6:]}")        # 1符号 5指数 10尾数(布局就和 FP32 不同了)






































