import torch


def quantize_symmetric(x):
    scale=x.abs().max()/127.0
    x_int8=torch.round(x/scale)
    x_int8=x_int8.clamp(-128,127).to(torch.int8)
    return x_int8,scale

def dequantize_symmetric(x_int8,scale):
    return x_int8.to(torch.float32)*scale

def quantize_asymmetric(x):
    xmin,xmax=x.min(),x.max()
    scale=(xmax-xmin)/225.0
    zero_point=torch.round(-xmin/scale)-128
    x_int8=torch.round(x/scale+zero_point)
    x_int8=x_int8.clamp(-128,127).to(torch.int8)
    return x_int8,scale,zero_point

def dequantize_asymmetric(x_int8,scale,zero_point):
    return (x_int8.to(torch.float32)-zero_point)*scale

torch.manual_seed(0)
w=torch.randn(1000)*0.5

w_q,s=quantize_symmetric(w)
w_hat=dequantize_symmetric(w_q,s)
err=(w-w_hat).abs()
print(f"scale(每格宽度)   : {s.item():.6f}")
print(f"最大还原误差       : {err.max().item():.6f}  (理论上 ≤ scale/2 = {s.item()/2:.6f})")
print(f"平均还原误差       : {err.mean().item():.6f}")
print(f"原始存储           : {w.numel()*4} 字节 (FP32)")
print(f"量化后存储         : {w_q.numel()*1} 字节 (INT8) + 1个scale → 省 ~4x")







































