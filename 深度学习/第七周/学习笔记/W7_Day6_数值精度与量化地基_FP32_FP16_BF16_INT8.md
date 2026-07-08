# W7 Day6 · 数值精度与量化地基:FP32 / FP16 / BF16 / INT8——memory-bound 的第二把杀器

> **本笔记的唯一目标**:让你从"听说半精度能省显存"升级到**能从 IEEE 754 的比特布局讲起、量出本机真实的速度差与数值误差、并把"误差可控可解释"写成能跑的断言代码**。读完你要能讲清四件事——(1) FP16 / BF16 **同样 16 位,为什么一个易溢出、一个训练稳**?答案在"指数位 vs 尾数位"怎么分,要能画出比特布局、并用底层代码验证 BF16 就是"FP32 砍掉后 16 位";(2) 为什么"精度减半"对 **memory-bound** 的 decode 是**直接提速**、而且转换开销几乎免费——这是 Day1 Roofline / W6 §5.3 结论的直接兑现;(3) INT8 量化 `x_int8 = round(x/scale)+zero_point` 里 **scale / zero_point** 到底是什么、对称和非对称差在哪,要能手写 quant/dequant 并量出还原误差;(4) **(本篇灵魂,对接课题主线 4)** 为什么"量化必须配误差度量",以及"误差可控可解释"落到代码上就是 `allclose` / logits cosine / 端到端 PPL 这三级度量——这是 W6 §4.4"优化版 vs 朴素版数值对比"职业素养的正式升级。
>
> **串联**:这是 [W7 学习计划](./W7_学习计划_AI_Infra主线.md) **Day6**,直接对接小米课题**主线 4(验证:数值一致或误差可控可解释)**。承接 [Day1 Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md) 与 [W6 KV Cache §5.3](./人工智能(md)/W6_Day6_KV_Cache_两阶段_显存代价.md)——那里证明了 decode 是 **memory-bound(瓶颈在搬数据不在算)**,今天给出这个结论最直接的"提速药方":搬运量减半。也正式升级 [W6 §4.4](./人工智能(md)/W6_Day6_KV_Cache_两阶段_显存代价.md) 的"优化版必须和朴素版数值等价"——从 KV Cache 的 `torch.equal` 断言,升级到量化场景"允许有误差、但误差必须被度量和解释"。与 [Day4 融合](./W7_Day4_KernelLaunch开销与算子融合_巨核动机.md)并列:融合是 memory-bound 的第一把杀器(省访存的次数),量化是第二把(省每次访存的字节数)。
>
> **产出对齐**:本笔记正文即计划要求的 `tech_notes/precision_and_quantization.md`(仓库里叫这个名;桌面按 `W7_DayX` 惯例平铺)。配套实测脚本 `bench_precision.py` + FP32/FP16 速度对比 + 数值误差数据(allclose / cosine)。

---

## 0. 开篇:读完你要能不看资料答出来的问题

1. FP16 和 BF16 都是 16 位,**位布局**差在哪?为什么 BF16"范围和 FP32 一样大"却"精度更差"?一句话:省下来的位从哪儿挪到哪儿了?
2. 为什么**训练偏爱 BF16、而 KV Cache 常用 FP16**?(提示:训练怕的是"溢出",KV Cache 怕的是"精度不够")
3. 为什么"精度减半 = 直接给 decode 提速"?为什么说这个提速在 memory-bound 场景**几乎是免费的**?→ 接 Day1 / W6 §5.3
4. FP16 为什么会**溢出(overflow)/下溢(underflow)**?**loss scaling(损失缩放)** 是怎么用一个乘法救回下溢的梯度的?
5. INT8 量化里的 **scale(缩放因子)** 和 **zero_point(零点)** 各是什么?**对称 vs 非对称**量化差在哪、各自适合什么数据?
6. **(灵魂题)** 为什么"量化必须配误差度量"?课题验收里"数值一致或误差可控可解释"这句话,落到代码上是哪三行度量?

> 第 1、3、6 题是题眼。第 1 题你要能徒手画出三种格式的比特分布并说清"BF16 = FP32 截断";第 6 题你要能说出"allclose 管逐元素、cosine 管方向、PPL 管任务效果"这三级,并解释为什么单看 allclose 会骗人。

---

## 1. 问题背景:decode 卡在"搬数据",那就让"数据变小"

先把前几天的结论接上,你才知道量化在整张地图里补的是哪一块。

- **Day1(Roofline)** 算出 H100 的脊点 ≈ 295 FLOP/byte——一个算子每从显存搬 1 字节,得做够 295 次浮点运算才不被带宽拖累。
- **W6 §5.3** 证明:decode 阶段为生成 1 个 token,要把**整个模型权重 + 全部 KV Cache** 从显存(HBM)搬进计算单元,却只做很少的计算,算术强度远低于 295 → **铁定 memory-bound(受内存带宽限制)**。
- **Day4** 给了第一把杀器:**算子融合**——减少访存的**次数**(中间结果不落 HBM)。

今天补第二把杀器。既然瓶颈是"搬数据太慢",而"搬多少数据 = 张量元素个数 × 每个元素的字节数",Day4 已经从"个数"下手了(融合减少中间张量),今天从**"每个元素的字节数"**下手:

> **核心一句话**:一个数用 FP32 存要 **4 字节**,用 FP16/BF16 存只要 **2 字节**,用 INT8 只要 **1 字节**。在 memory-bound 的 decode 里,**搬运字节数减半,墙钟时间就几乎直接减半**——因为瓶颈是带宽,你把要搬的货砍掉一半,车程自然短一半。

**为什么说这个提速"几乎免费"?** 这是最关键、也最反直觉的一点,必须讲透:

- memory-bound 意味着**算力单元大部分时间在空转、等数据**(Day1 结论)。也就是说 GPU 的"算"这条腿本来就**过剩、闲着**。
- 用低精度,数据搬得快了,原本闲着的算力稍微多干一点(比如把 FP16 转成 FP32 再算、或用 Tensor Core 直接吃 FP16)——**这点额外的算,是在原本空转的时间里顺手做掉的,不占用瓶颈资源**。
- 类比:一条**独木桥(带宽)**是瓶颈,桥两头的**搬运工(算力)**闲得发慌。现在你把货物打包得更小(低精度),过桥快了一倍;打包/拆包这点活,搬运工反正闲着,顺手就干了,不额外花时间。**这就是"转换开销几乎免费"的本质——它花的是过剩资源。**

> **反过来提醒**:如果一个负载是 **compute-bound(受算力限制)**,比如 prefill 的大矩阵乘、或训练,那"低精度省带宽"的收益就没这么直接了——那时收益主要来自 **Tensor Core 对低精度有更高的算力吞吐**(FP16/BF16 的 TFLOPS 是 FP32 的好几倍),是另一条增益路径。所以"低精度为什么快"在两种 bound 下**理由不同**,别混:memory-bound 靠**省带宽**,compute-bound 靠**Tensor Core 更高吞吐**。

## 2. 核心一:三种浮点的比特布局——同样 16 位,命运不同

要真正搞懂 FP16 和 BF16 的区别,必须下沉到**一个浮点数在内存里到底长什么样**。这是全篇地基。

### 2.1 是什么:浮点数 = 符号 × 尾数 × 2^指数

**浮点数(floating-point number)** 的存储,本质是把科学计数法搬进二进制。一个数被拆成三段比特:

- **sign(符号位)**:1 位,0 正 1 负。
- **exponent(指数位)**:决定**动态范围**(能表示多大到多小),相当于科学计数法里 10^k 的那个 k。**指数位越多,能表示的数的范围越大。**
- **mantissa / fraction(尾数位 / 小数位)**:决定**精度**(同一个量级内能分得多细),相当于 1.2345 里"2345"那几位有效数字。**尾数位越多,数越精确。**

一句话记死:**指数管"够不够大"(范围),尾数管"够不够准"(精度)。** 16 位就那么多,给了指数就少给尾数,这就是 FP16 和 BF16 分道扬镳的全部原因。

### 2.2 三种格式对照:同样是"缩水",缩的地方不一样

| 格式 | 中文 | 符号 | 指数 | 尾数 | 总位数 | 动态范围(约) | 相对精度 | 一句话定位 |
|---|---|---|---|---|---|---|---|---|
| **FP32** | 单精度 | 1 | 8 | 23 | 32 | 1e−38 ~ 3e38 | 高 | 训练/推理的"黄金标准",4 字节 |
| **FP16** | 半精度 | 1 | **5** | **10** | 16 | **6e−5 ~ 65504** | 中 | 范围小**易溢出**,精度尚可,2 字节 |
| **BF16** | 脑浮点 | 1 | **8** | **7** | 16 | **1e−38 ~ 3e38** | 低 | 范围**=FP32**,精度差,训练稳,2 字节 |

盯着这张表看两组对比,今天的一切都从这里长出来:

**① FP16 vs FP32(同样是浮点,FP16 为什么"易溢出"):** FP16 只有 5 个指数位,能表示的最大数只到 **65504**。深度学习里一个矩阵乘的中间结果、或梯度,轻轻松松就超过 6 万 → **溢出成 `inf`(无穷)**;而很小的梯度(比如 1e−8)小于 FP16 能表示的最小正数 6e−5 → **下溢成 0**,梯度直接消失。这就是 §4 要讲的 FP16 训练两大杀手。

**② BF16 vs FP16(同样是 16 位,BF16 凭什么训练稳):** 这是今天第一个"啊哈"点。BF16 的设计哲学是:

> **"我把 FP16 从尾数里省出来的 3 位,全部还给指数。"** FP16 是 5 指数 + 10 尾数;BF16 是 8 指数 + 7 尾数。**BF16 的 8 个指数位,和 FP32 一模一样**——所以 BF16 的动态范围**等于 FP32**,根本不会像 FP16 那样一算就溢出。代价是尾数只剩 7 位,精度比 FP16 还差。

**为什么这个取舍对训练是对的?** 训练时最怕的是**溢出/下溢导致 `NaN`、梯度消失**(范围问题),而不太怕"某个数不够精确"(精度问题)——因为梯度下降本身就是带噪声的迭代,少几位有效数字无所谓,但一个 `inf` 能让整个训练崩掉。所以 **BF16 用"精度换范围"正好对症**:范围管够(不崩),精度差点(能忍)。这就是知识点里那句"训练偏爱 BF16"的根。

### 2.3 底层代码:BF16 其实就是"FP32 砍掉后 16 位"

BF16 有个极优雅的性质,理解它你就彻底懂了 BF16 和 FP32 的关系:**因为 BF16 的符号位、指数位布局和 FP32 完全相同(1+8),BF16 就是把一个 FP32 数的高 16 位直接截取下来、扔掉低 16 位尾数。** 转换几乎零成本(连指数都不用重算),这也是硬件爱它的原因之一。

```python
# 环境: PyTorch>=1.10 + 支持 BF16 的 CPU/GPU。目的: 亲眼看清三种格式的"缩水"发生在哪几位。
import torch, struct

def bits_of(x_tensor):
    """把一个标量张量的原始比特打印成二进制字符串(按其 dtype 的位宽)。"""
    # view 成整数类型, 才能拿到底层比特; 这是"看内存原始字节"的标准手法
    if x_tensor.dtype == torch.float32:
        b = x_tensor.view(torch.int32).item() & 0xFFFFFFFF; w = 32
    elif x_tensor.dtype == torch.float16:
        b = x_tensor.view(torch.int16).item() & 0xFFFF; w = 16
    elif x_tensor.dtype == torch.bfloat16:
        b = x_tensor.view(torch.int16).item() & 0xFFFF; w = 16
    return format(b, f'0{w}b')

val = 1.5
f32 = torch.tensor(val, dtype=torch.float32)
f16 = f32.half()                 # 转 FP16
bf16 = f32.bfloat16()            # 转 BF16

s32 = bits_of(f32)
print(f"FP32 : {s32[0]} {s32[1:9]} {s32[9:]}")     # 1符号 8指数 23尾数
sb = bits_of(bf16)
print(f"BF16 : {sb[0]} {sb[1:9]} {sb[9:]}")        # 1符号 8指数 7尾数
# ★ 关键验证: BF16 的 16 位, 恰好等于 FP32 的高 16 位!
print(f"FP32 高16位 == BF16 ? {s32[:16] == sb}")   # 预期 True (1.5 这类整洁的数)
sh = bits_of(f16)
print(f"FP16 : {sh[0]} {sh[1:6]} {sh[6:]}")        # 1符号 5指数 10尾数(布局就和 FP32 不同了)
```

**读这段的收获**:你会亲眼看到 BF16 的比特串 = FP32 的前 16 位(指数段完全对齐),而 FP16 的指数段只有 5 位、布局和 FP32 不一样(所以 FP16↔FP32 转换要真的重排指数,不像 BF16 那样一刀切)。**"BF16 = 截断的 FP32"这个直觉,比背'8指数7尾数'有用一百倍。**

## 3. 核心:为什么"精度减半"是 memory-bound 的大杀器

这一节是本篇和你课题、和 Day1/Day4 咬合最紧的地方。**为什么把 FP32 换成 FP16 能直接给 decode 提速?** 答案不在"算得快",而在"搬得少"。

### 3.1 先接上 W6 §5.3 / Day1 的结论:decode 卡在搬运,不卡在计算

回忆你已经证明过的两件事:

- **W6 §5.3**:decode(逐 token 生成)是 **memory-bound**——瓶颈是"把权重和 KV Cache 从显存搬到计算单元的速度",算力单元大部分时间在**等数据**、空转。
- **Day1 Roofline**:H100 的脊点(ridge point)≈ 295 FLOP/byte,意思是"每搬 1 字节要做够 295 次浮点运算才不浪费带宽"。decode 每生成 1 个 token,要把**整个模型的权重 + 全部 KV Cache** 搬一遍,却只做那一个 token 的少量计算 → 算术强度远低于 295 → **铁定 memory-bound**。

把这两句话合起来,decode 一步的耗时可以粗略写成:

```
decode 单步耗时 ≈ (要搬运的字节数) / (显存带宽)
              ≈ (模型权重字节 + KV Cache 字节) / HBM带宽
```

**注意这个式子里根本没有"算力"**——因为算力过剩、不是瓶颈,时间几乎完全由"搬多少字节"决定。

### 3.2 那么关键推论就来了:字节减半 = 时间减半

既然耗时 ≈ 字节数 / 带宽,而带宽是硬件定死的,那**唯一能动的就是"字节数"**。

> **FP32 → FP16,每个数从 4 字节变 2 字节,要搬的总字节数直接砍半 → decode 单步耗时理论上直接减半、吞吐翻倍。** 这就是"精度减半是 memory-bound 大杀器"的全部道理——它不是让 GPU 算得更快,而是**让 GPU 少等一半时间**。

**类比**:decode 像用小水桶(计算单元)去一个远处的大水池(显存)一桶一桶运水,你手脚再快(算力再强)也没用,瓶颈是**跑这段路的时间**。FP32 好比每桶装的是"体积虚胖的水"(4 字节/数),FP16 是把同样的信息压成"一半体积"(2 字节/数)——**跑的趟数不变,但每趟搬的字节少一半,总搬运时间减半**。而你手脚(算力)本来就没闲着等水的份,压缩解压这点手上功夫(转换)顺手就做了,几乎不额外花时间。

### 3.3 为什么"转换几乎免费":算力本来就是过剩的

这里补一个容易忽略、但很关键的点。有人会担心:"把 FP16 读进来还要转成 FP32 算、算完再转回去,这不也要花时间吗?"

答案是:**在 memory-bound 场景,这点转换开销几乎白送。** 因为:

1. **算力过剩**:decode 时计算单元大把时间在空等数据(§3.1),你让它顺手做几个 FP16↔FP32 的格式转换,**用的正是它本来空转的那些周期**,不占用"搬运"这个真正的瓶颈。
2. **现代 GPU 有硬件级低精度支持**:H100 的 **Tensor Core(张量核心,专做矩阵乘的硬件单元)** 原生吃 FP16/BF16/FP8,连"转换"都省了,直接用低精度算,算力反而**翻倍甚至更多**(FP16 Tensor Core 算力 ≈ FP32 的 2 倍以上)。

所以低精度在 memory-bound 场景是"**稳赚**":省了搬运(主要收益)+ 可能还加速了计算(附赠)+ 省了一半显存(KV Cache 能存更长序列/更大 batch)。**三重好处,几乎零代价**——这就是为什么工业界 LLM 推理**默认**就是 FP16/BF16,FP32 推理反而罕见。

### 3.4 一张账:KV Cache 用 FP16 省下的显存,直接换成更长上下文

把 §3 落到你最熟的 KV Cache(W6 全篇)上,算一笔具体的账,你会对"省显存"有体感:

```
一个 token 的 KV Cache 显存 = 2(K和V) × 层数 × 注意力头数 × head_dim × 精度字节数
以 Llama-2-7B 为例: 2 × 32层 × 32头 × 128 × 精度字节
  FP32: 2×32×32×128×4 = 2 MB / token
  FP16: 2×32×32×128×2 = 1 MB / token   ← 直接省一半
```

**这一半省下来意味着什么?** 同样一张 80GB 的 H100,KV Cache 用 FP16 而非 FP32,**能缓存的总 token 数翻倍**——要么支持**两倍长的上下文**,要么在 continuous batching 里**多塞一倍的并发请求**。这就是知识点里"**KV Cache 常用 FP16**"的根本原因:KV Cache 是 decode 阶段访存的大头,把它减半,memory-bound 的病根直接缓解一半,还顺带把最贵的显存资源省出来给吞吐。

> **工业锚点(对接你课题主线 1/4)**:vLLM、TensorRT-LLM 默认把 KV Cache 存成 FP16/BF16,近年更激进的直接上 **FP8 KV Cache**(每 token 再砍半)。你的小米课题要"降低访存/中间张量",KV Cache 精度就是最直接的一个旋钮——但**动它就必须配 §5 的误差度量**,证明"误差可控可解释",否则不敢上线。

## 4. FP16 为什么会"溢出/下溢"?loss scaling 又是什么?

§2 说 FP16 只有 5 位指数、动态范围小。这一节把"范围小"会闯什么祸讲透,并解释训练界为什么发明了 **loss scaling** 这根拐杖——以及为什么 BF16 出来后这根拐杖就不太需要了。

### 4.1 是什么:overflow(上溢)和 underflow(下溢)

先给两个词的直观定义(都是"数值超出了 FP16 能表示的范围"):

- **overflow(上溢)**:一个数**太大**,超过了 FP16 能表示的最大值(≈ 65504),结果变成 **`inf`(无穷大)**。一旦出现 inf,后面的运算基本全废(inf 参与运算会传染成 inf 或 `nan`)。
- **underflow(下溢)**:一个数**太小**(绝对值),小到 FP16 最小正规数(≈ 6.1e-5)都表示不了,结果被**归零**。数值本身不大不小时没事,但"本该是很小的非零数"变成 0,信息就丢了。

**类比**:FP16 像一把**量程很窄的秤**(比如只能称 0.01 克到 65 公斤)。称一头大象(大梯度)→ 爆表显示"∞"(overflow);称一粒灰尘(小梯度)→ 秤根本没反应、显示 0(underflow)。BF16 那把秤量程和 FP32 一样宽(能称 10⁻³⁸ 到 10³⁸),大象灰尘都称得出,只是**刻度粗**(精度低)——这正是 §2.3"BF16 拿精度换范围"的后果在训练里的体现。

### 4.2 为什么偏偏训练容易中招,推理没那么怕

关键区别在**梯度**:

- **推理(你的主场)**:只有前向传播,数值大多是激活值,范围相对温和;而且 KV Cache、权重这些数值本来就不极端 → FP16 一般够用,溢出风险低。
- **训练**:有反向传播,**梯度(gradient)** 的数值范围极其夸张——深层网络里梯度可能非常小(小到 FP16 下溢归零,这一层就"学不动"了),偶尔又因为某个异常样本突然很大(上溢成 inf,整个训练 crash)。这就是训练比推理更怕 FP16 的根本原因。

### 4.3 老办法:loss scaling(损失缩放)——把小梯度"托举"进 FP16 的量程

在 BF16 普及前,大家想用 FP16 训练省显存,又要对付梯度下溢,于是发明了 **loss scaling**。

> **loss scaling(损失缩放)是什么**:反向传播前,先把 loss(损失)乘一个大常数 S(比如 1024 或动态调整),于是链式法则下**所有梯度都被同比放大 S 倍**——那些本来会下溢归零的小梯度,被"托举"回 FP16 能表示的正规范围内;等梯度算完、要更新权重时,再把梯度**除回 S** 还原。

**为什么这么做能成立(底层直觉)**:梯度下溢的本质是"数值落在了 FP16 量程的地板以下"。loss scaling 不改变梯度之间的**相对大小**(全体同乘 S),只是把它们整体**平移进量程**——就像用显微镜把太小看不清的东西放大 S 倍看清楚,记完数再按比例缩回去。关键前提:**放大后不能反而上溢**,所以工业实现用的是 **dynamic loss scaling(动态损失缩放)**——S 从一个大值开始,一旦检测到梯度出现 inf/nan 就把 S 减半、跳过这步更新;连续若干步没出事就试着把 S 翻倍。PyTorch 的 `torch.cuda.amp.GradScaler` 就是干这个的。

### 4.4 底层代码:loss scaling 到底改了哪一行

```python
# 环境: PyTorch>=1.6 + CUDA。这是 FP16 混合精度训练的工业标准写法(AMP)。
# 对比看: 加了 scaler 的 4 行, 就是 loss scaling 的全部实现。
import torch
from torch.cuda.amp import autocast, GradScaler

model = ...      # 你的模型
optimizer = ...  # 你的优化器
scaler = GradScaler()   # ← dynamic loss scaling 的管理器(自动调 S、自动跳过 inf 步)

for x, y in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.float16):   # ← 前向自动用 FP16 算(该转FP32的算子它自己转)
        loss = loss_fn(model(x), y)

    # —— 下面 4 行就是 loss scaling 的核心 ——
    scaler.scale(loss).backward()   # ① loss × S 再反传 → 梯度同比放大 S, 躲开下溢
    scaler.step(optimizer)          # ② 内部先把梯度 ÷ S 还原; 若发现 inf/nan 则本步跳过不更新
    scaler.update()                 # ③ 根据这步有没有 inf, 动态调整下一步的 S(减半/翻倍)
    # 若不用 FP16 而用 BF16(dtype=torch.bfloat16), 通常整段 scaler 都不需要 → 见 4.5
```

**读这段的关键**:`autocast` 负责"前向用低精度算",`GradScaler` 才是 loss scaling 本体。**注意 `scaler.step` 内部会先除回 S 再更新权重**——所以你的学习率不用改,scaling 对优化器是透明的。这套组合叫 **AMP(Automatic Mixed Precision,自动混合精度)**,是 FP16 训练的行业标配。

### 4.5 新办法:BF16 让 loss scaling 基本退休(呼应知识点"训练偏爱 BF16")

现在把 §2.3 的伏笔收掉。为什么现在大模型训练几乎清一色 **BF16**,而 loss scaling 越来越少见?

> 因为 **BF16 的动态范围和 FP32 一样大(8 位指数)**,梯度再小也基本不下溢、再大也基本不上溢——**根本不需要 loss scaling 这根拐杖**。代价是 BF16 精度低(只有 7 位尾数),但训练对精度不敏感(梯度本来就带噪声,SGD 是个"模糊"的过程),对范围极其敏感——所以**用范围换精度的 BF16,恰好卡在训练的需求上**。这就是知识点那句"训练偏爱 BF16"的完整因果:**训练怕的是溢出(范围问题),不是不够精确(精度问题),而 BF16 正好保范围。**

反过来,**KV Cache 用 FP16** 而不用 BF16,是因为 KV Cache 里的数值范围温和、不会溢出,此时**宁愿要 FP16 那多出来的 3 位尾数(精度)**——同样 2 字节,FP16 在"范围够用"的前提下比 BF16 更精确。**一句话记死:范围有风险选 BF16,范围没问题选 FP16。**

## 5. INT8 量化:把浮点"塞进"8 个格子里

FP16/BF16 还是浮点,只是位数少。**量化(quantization)** 更激进:直接把浮点数映射成**整数**(INT8,8 位整数,只能表示 -128~127 这 256 个值)。省得更狠(FP32→INT8 是 4 倍),但引入的误差也更大——这就是为什么 §6 的误差度量对量化**生死攸关**。

### 5.1 是什么:量化就是"用一把尺子把浮点范围切成 256 格"

> **quantization(量化)是什么**:把连续的浮点数值,用一个**线性映射**近似成有限个整数。INT8 量化就是把一段浮点范围(比如权重的 [-2.5, 2.5])**均匀切成 256 格**,每个浮点数**四舍五入到最近的格子编号**(一个 INT8 整数)。用的时候再按格子宽度**还原**回近似浮点值。

**类比**:量化像**把一段连续的温度用整数刻度的温度计去读**。真实温度是 23.7°C(浮点),温度计只有整数刻度 → 读成 24(整数)。你丢了 0.3 度的精度,但换来"只需记一个整数"的省事。**格子宽度(scale)** 就是"温度计每格代表多少度"——格子越细(range 越小),读数越准;格子越粗,越省但越不准。

### 5.2 底层公式:scale 和 zero-point 到底是什么

知识点给的公式:`x_int8 = round(x_fp32 / scale) + zero_point`。逐个拆:

- **scale(缩放因子)**:一个浮点数,表示**"每个整数格子代表多大的浮点跨度"**。它把浮点范围压进整数范围。比如浮点范围 [-2.5, 2.5] 共 5.0 宽,要塞进 256 格,则 `scale ≈ 5.0 / 255 ≈ 0.0196`——每格约 0.02。**scale 越小,量化越精细,但能覆盖的范围越窄。**
- **zero-point(零点)**:一个整数,表示**"浮点里的 0,对应到整数轴上的哪一格"**。它是为了处理**不对称**的浮点范围(比如 ReLU 后的激活全是正数 [0, 6])——这时把整数轴平移一下,才能不浪费格子。

反量化(dequantize,还原)公式是量化的逆:`x_fp32 ≈ scale × (x_int8 - zero_point)`。注意那个 **`≈`**——**还原回来的值和原值不完全相等**,差的那点就是**量化误差(quantization error)**,它 ≤ 半个格子宽(scale/2)。这个"约等于"就是 §6 要死死盯住的东西。

### 5.3 对称 vs 非对称:zero-point 等不等于 0

这是知识点问的核心区别,用一张图理解:

- **对称量化(symmetric,zero_point=0)**:假设浮点范围**关于 0 对称**,比如 [-2.5, 2.5]。直接让浮点 0 对应整数 0,不用平移。`scale = max(|x|) / 127`。**优点**:公式简单(反量化就一个乘法,没有减法),硬件算得快。**适用**:权重——训练出来的权重通常近似关于 0 对称分布。
- **非对称量化(asymmetric,zero_point≠0)**:浮点范围**不对称**,比如 ReLU 后的激活 [0, 6](全正)。若还硬用对称量化,负半轴那 128 个格子全浪费了,精度腰斩。非对称量化把范围 [min, max] 整体映射到 [-128, 127],`scale = (max-min)/255`,`zero_point` 负责把浮点的 min 对齐到整数 -128。**优点**:不浪费格子,精度高。**代价**:反量化多一步减法(减 zero_point),稍慢。**适用**:激活——尤其 ReLU/GELU 后单边分布的。

> **一句话记法**:**权重对称、激活非对称**是最常见的工业默认(比如 PyTorch 的 `qint8` 权重 + `quint8` 激活)。理由就是上面:权重分布对称、激活常单边。

### 5.4 底层代码:手写一遍 quantize / dequantize,亲眼看误差

```python
# 环境: PyTorch>=1.10(纯 CPU 可跑, 不需要 GPU)。
# 目的: 手写对称/非对称 INT8 量化, 看还原误差从哪来、有多大。
#       这是理解所有量化库(bitsandbytes/AWQ/GPTQ)的地基。
import torch

def quantize_symmetric(x):
    # 对称量化: 适合权重(关于0对称)。zero_point 恒为 0。
    scale = x.abs().max() / 127.0          # 用绝对值最大值定 scale, 保证不溢出 [-127,127]
    x_int8 = torch.round(x / scale)        # 核心公式: 除以 scale 再四舍五入 → 落到整数格
    x_int8 = x_int8.clamp(-128, 127).to(torch.int8)  # clamp 防越界(round 可能到 128)
    return x_int8, scale

def dequantize_symmetric(x_int8, scale):
    return x_int8.to(torch.float32) * scale   # 反量化: 乘回 scale(对称无 zero_point, 没有减法)

def quantize_asymmetric(x):
    # 非对称量化: 适合激活(如 ReLU 后单边分布)。zero_point 负责平移。
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin) / 255.0                      # 用真实 min/max 定格子宽度
    zero_point = torch.round(-xmin / scale) - 128      # 让浮点 xmin 对齐到整数 -128
    x_int8 = torch.round(x / scale + zero_point)
    x_int8 = x_int8.clamp(-128, 127).to(torch.int8)
    return x_int8, scale, zero_point

def dequantize_asymmetric(x_int8, scale, zero_point):
    return (x_int8.to(torch.float32) - zero_point) * scale  # 先减 zero_point 再乘 scale

# ---- 实测: 对一段权重量化再还原, 看误差 ----
torch.manual_seed(0)
w = torch.randn(1000) * 0.5      # 模拟一层权重, 近似对称分布

w_q, s = quantize_symmetric(w)
w_hat = dequantize_symmetric(w_q, s)   # 还原
err = (w - w_hat).abs()
print(f"scale(每格宽度)   : {s.item():.6f}")
print(f"最大还原误差       : {err.max().item():.6f}  (理论上 ≤ scale/2 = {s.item()/2:.6f})")
print(f"平均还原误差       : {err.mean().item():.6f}")
print(f"原始存储           : {w.numel()*4} 字节 (FP32)")
print(f"量化后存储         : {w_q.numel()*1} 字节 (INT8) + 1个scale → 省 ~4x")
```

**预期现象**:最大误差确实 ≤ scale/2(半个格子),这印证了 §5.2 那句"误差 ≤ 半格宽"。**这段代码就是所有量化的原子**——工业库(bitsandbytes 的 LLM.int8()、GPTQ、AWQ)的复杂之处不在这个公式,而在 §8 会讲的"**怎么处理离群值、怎么选 scale**",公式本身就是上面这几行。

## 6. 【本篇灵魂】数值一致性:量化必配误差度量

这一节是整篇和你**小米课题主线 4** 最直接相关的部分,也是知识点里标注"**最重要**"的一环。前面讲的都是"怎么省",这一节讲"**怎么证明省得起**"——量化必然引入误差,一个"快但错"的量化模型**一文不值**。

### 6.1 为什么"量化必须配误差度量"

回想 §5:量化后 `x_fp32 ≈ scale × (x_int8 - zp)`,那个 `≈` 就是误差的来源。问题是:**这个误差,会不会累积到让模型输出错的结果?**

答案是"看情况"——有时无害,有时致命。而"看情况"这三个字,在工业上是**不允许**的。你的课题验收指标白纸黑字写着"**数值一致或误差可控可解释**",PO 不会接受"我觉得差不多"。所以:

> **铁律**:任何一个"优化版"(量化 / 融合 / 巨核 / KV Cache),都**必须**配一个能量化"它和原版差多少"的**误差度量(error metric)**,并给出这个数字**在可接受范围内**的证据。这不是加分项,是 AI Infra 的**基本职业素养**——直接延续你 [W6 Day6 §4.4](./人工智能(md)/W6_Day6_KV_Cache_两阶段_显存代价.md)"优化版必须和朴素版数值对比"那条,只是从 KV Cache 换成了量化。

**类比**:量化像**给一份法律合同做"精简版"**。精简版读起来快(省显存/提速),但你必须证明"**关键条款一字没改**"(误差可控)。你不能拿着精简版说"我大概看了下,意思差不多"——你要**逐条比对**,并出具"差异清单"(误差报告)。§6.2 讲的三把尺子,就是三种精度的"比对方法"。

### 6.2 三把尺子:从最严到最宽

课题和知识点点名的三个度量,其实是**三个不同粒度**,由近及远、由严到宽:

| 度量 | 中文 | 看的是什么 | 严格度 | 什么时候用 |
|---|---|---|---|---|
| **`allclose`** | 逐元素接近 | 每个数字差多少 | 最严(逐元素) | 融合/KV Cache 这种"理论上应该几乎相等"的 |
| **logits cosine** | 输出向量方向相似度 | 输出的"整体形状"变没变 | 中(看方向不看绝对值) | 量化这种"数值必变、但决策不该变"的 |
| **end-to-end PPL** | 端到端困惑度 | 模型语言能力掉了多少 | 最宽(只看最终任务) | 上线前的终极验收 |

逐个拆:

**① `torch.allclose(a, b, rtol, atol)`——逐元素接近**

> **`allclose` 是什么**:逐个元素检查 `|a - b| ≤ atol + rtol × |b|` 是否处处成立。`atol`(absolute tolerance,绝对容差)管"接近 0 时允许差多少",`rtol`(relative tolerance,相对容差)管"数值大时允许差百分之几"。全部满足才返回 True。

这是**最严**的尺子。对 KV Cache、算子融合这类"数学上应该恒等、只有浮点舍入误差"的优化,就该用它(呼应 W6 §4.4 用 `torch.equal`/`allclose` 断言)。但对**量化**——数值必然变化明显——`allclose` 几乎必然 False,**不适合当主度量**,只适合看"误差有没有大到离谱"。

**② logits cosine similarity——输出方向相似度**

> **cosine similarity(余弦相似度)是什么**:把两个向量看成两支箭,算它们**夹角的余弦**。值域 [-1, 1],=1 表示方向完全一致,=0 表示垂直(毫不相关)。它**只看方向、不看长度**——这正是量化想要的:量化让每个 logit 数值都变了(长度变了),但只要**相对大小关系没变**(方向没变),`argmax` 选出的 token 就不变,模型行为就一致。

> **logits(对数几率)是什么**:模型最后一层输出的、**过 softmax 之前**的原始分数向量,每个词表 token 一个分数。谁的 logit 最大,就最可能是下一个 token。之所以比"最终输出的文本"更适合做度量,是因为它是**连续值**——能看出"差了多少",而文本是离散的,要么一样要么不一样,损失了细节。

对量化,**logits cosine 是最贴切的主度量**:它容忍数值变化,只揪"决策方向"有没有跑偏。工业阈值通常要求 **cosine > 0.99** 甚至 0.999。

**③ end-to-end PPL(perplexity,困惑度)——终极任务度量**

> **perplexity(困惑度,PPL)是什么**:衡量语言模型"预测下一个词有多不确定"的指标,数学上 `PPL = exp(平均交叉熵)`。**越低越好**——PPL=10 直观理解为"模型每一步平均在约 10 个词里纠结"。它是**端到端**的:不看中间张量,只看"量化后模型的语言能力掉了多少"。(这正好接上你 W7 数学任务里的**交叉熵/信息论**——PPL 就是交叉熵取指数。)

这是**最宽、也最有说服力**的尺子。量化 PR 上线前的终极验收就是它:在一个标准数据集(如 WikiText)上跑 FP32 和量化版,比 PPL。工业容忍度通常是 **PPL 上升 < 1%**。它是你课题交付物"真实业务模型性能对比报告"里必然要有的一栏。

### 6.3 底层代码:一次性把三把尺子都量出来

```python
# 环境: PyTorch>=2.0。GPU 更贴近真实(FP16 加速), CPU 也能跑度量逻辑。
# 目的: 对同一个模型的 FP32 输出 vs 低精度输出, 同时算三把尺子。
#       这就是课题"误差可控可解释"落到代码上的样子——一份可复现的误差报告。
import torch
import torch.nn.functional as F

@torch.no_grad()
def error_report(logits_ref, logits_test, name="test"):
    """logits_ref: FP32 基准输出; logits_test: 低精度/量化版输出。
       两者形状相同 [batch, vocab] 或 [batch, seq, vocab]。"""
    ref = logits_ref.float().flatten(end_dim=-2)   # 统一成 [N, vocab], 且转 FP32 再比
    test = logits_test.float().flatten(end_dim=-2) # 关键: 比之前都升到 FP32, 否则比的是"两个有误差的数"

    # 尺子①: 逐元素接近(量化下通常 False, 只看误差量级)
    max_abs = (ref - test).abs().max().item()
    allclose = torch.allclose(ref, test, rtol=1e-2, atol=1e-2)

    # 尺子②: logits 余弦相似度(量化的主度量, 看决策方向)
    cos = F.cosine_similarity(ref, test, dim=-1).mean().item()

    # 尺子③(近似): top-1 一致率——argmax 选的 token 有多少比例没变
    #   这是 PPL 的"轻量替身", 不需要跑完整数据集也能立刻看到决策一致性
    top1_agree = (ref.argmax(-1) == test.argmax(-1)).float().mean().item()

    print(f"[{name}] 最大逐元素误差 : {max_abs:.4e}")
    print(f"[{name}] allclose       : {allclose}  (量化下 False 正常)")
    print(f"[{name}] logits cosine  : {cos:.6f}  (>0.99 才算方向一致)")
    print(f"[{name}] top-1 一致率   : {top1_agree:.4f}  (=1.0 表示每步选的 token 都没变)")
    return cos, top1_agree

# ---- 用法示例: 同一输入, FP32 vs FP16 ----
torch.manual_seed(0)
lin = torch.nn.Linear(512, 5000)             # 模拟输出层: hidden→vocab
x = torch.randn(8, 512)

logits_fp32 = lin(x)                          # 基准
logits_fp16 = lin.half()(x.half()).float()    # FP16 版, 再升回 FP32 做公平比较
error_report(logits_fp32, logits_fp16, "FP16")
```

**读这段要抓的两个工业细节:**

1. **比之前必须都升回 FP32**(`ref.float()`)。否则你是在"两个都有误差的低精度数"之间比,量出来的差是假的。这是新手量误差最常见的错。
2. **top-1 一致率是 PPL 的"平民替身"**。完整 PPL 要跑数据集、要真实权重,一时半会儿在 nanoGPT toy 上不方便;但 top-1 一致率(argmax 选的 token 变没变)**立刻能算**,且直觉极强——它就是"生成的文本会不会变"的直接预测。工业里快速回归常先看它,再上完整 PPL。

## 7. 动手实测:nanoGPT FP32 vs FP16 速度差 + 数值误差

知识点要求的动手:对 nanoGPT 跑 FP32 vs FP16 推理,实测**速度差 + 数值误差**。

> **好消息(和 Day5 相反)**:FP16 推理**不依赖 Triton**——`.half()` / `torch.autocast` 是 PyTorch + cuDNN/cuBLAS 原生支持的,**你本机 Windows RTX 5060 就能跑**(RTX 5060 有 FP16 Tensor Core)。所以这个实验今天在本机做即可,不用等 H100。这也是量化"部署友好"的一个侧证:它比手写 kernel 的门槛低得多。

### 7.1 完整脚本 `bench_precision.py`

```python
# ============================================================
# bench_precision.py
# 目的: 同一个 GPT block, FP32 vs FP16, 量 (1)速度差 (2)数值误差
# 运行环境: Windows/Linux + NVIDIA GPU(本机 RTX 5060 即可), PyTorch>=2.0
#   python bench_precision.py
# ============================================================
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
assert torch.cuda.is_available(), "要 GPU 才有 FP16 加速; 纯 CPU 上 FP16 反而可能更慢"
dev = "cuda"

# ---- 一个"像 nanoGPT 一层"的 block: attention + FFN, 参数量够大才看得出访存收益 ----
class Block(nn.Module):
    def __init__(self, d=1024, heads=16, hidden=4096):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)          # 融合 QKV(呼应 W6 Day3)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)
        self.heads = heads
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        # reshape 成多头, 用 PyTorch 自带的 flash attention(它内部就用 FP16 累加优化)
        q, k, v = [t.view(B, T, self.heads, C//self.heads).transpose(1,2) for t in (q,k,v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1,2).reshape(B, T, C))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))   # FFN 残差
        return x

model = Block().to(dev).eval()
# prefill 场景: 一次喂一段序列(T=512), 权重大 → 最能体现"搬运量减半"
x_fp32 = torch.randn(1, 512, 1024, device=dev)

def bench(model, x, warmup=30, iters=100):
    for _ in range(warmup): model(x)          # 热身: 躲首次 kernel 选择/显存池初始化
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): model(x)
    e.record(); torch.cuda.synchronize()      # 异步! 必须 sync 后再读时间
    return s.elapsed_time(e) / iters           # ms/iter

with torch.no_grad():
    # --- baseline: FP32 ---
    t_fp32 = bench(model, x_fp32)
    out_fp32 = model(x_fp32)

    # --- FP16: 权重和输入都转半精度 ---
    model_fp16 = Block().to(dev).eval()
    model_fp16.load_state_dict(model.state_dict())  # 同一套权重, 保证公平
    model_fp16 = model_fp16.half()                  # 权重 → FP16
    x_fp16 = x_fp32.half()                           # 输入 → FP16
    t_fp16 = bench(model_fp16, x_fp16)
    out_fp16 = model_fp16(x_fp16).float()            # 输出升回 FP32 再比误差

# ---- 速度 ----
print(f"FP32: {t_fp32:.4f} ms/iter  (baseline)")
print(f"FP16: {t_fp16:.4f} ms/iter  ({t_fp32/t_fp16:.2f}x faster)")
# ---- 误差(复用 §6.3 的三把尺子) ----
diff = (out_fp32 - out_fp16).abs()
cos = F.cosine_similarity(out_fp32.flatten(1), out_fp16.flatten(1), dim=-1).mean()
print(f"最大逐元素误差: {diff.max().item():.4e}")
print(f"平均逐元素误差: {diff.mean().item():.4e}")
print(f"输出 cosine   : {cos.item():.6f}  (>0.999 说明 FP16 几乎无损)")
```

### 7.2 预期现象 + 怎么解读(数量级示意,别死记具体值)

- **速度**:FP16 通常有 **1.3×~2×** 加速(取决于 GPU 是否吃满 Tensor Core、张量够不够大)。加速来源正是 §3 说的两笔:**搬运量减半**(memory-bound 部分直接受益)+ **Tensor Core FP16 吞吐更高**(compute 部分也受益)。张量太小时加速不明显,甚至更慢——因为 launch 开销占了大头(Day4 结论),FP16 省的那点访存被淹没。
- **误差**:FP16 的输出 cosine 通常 **> 0.999**,最大逐元素误差在 `1e-2 ~ 1e-1` 量级。**这就是"误差可控可解释"的一份具体证据**:你能指着 cosine=0.9995 说"FP16 推理决策方向几乎没变,可以上"。
- **如果 cosine 掉到很低(如 < 0.99)**:大概率某处发生了 **FP16 溢出**(§4)——比如 attention score 没做数值稳定处理、或某个中间激活值超了 65504。这时该上 **BF16**(`x.bfloat16()`,范围和 FP32 一样,基本不溢出)或 autocast(混合精度,敏感算子留 FP32)。

### 7.3 进阶:一行开混合精度(工业最常用姿势)

上面手动 `.half()` 是为了教学看清"全 FP16"。**工业推理里更常用 `torch.autocast`**——它自动决定"哪些算子用 FP16、哪些留 FP32":

```python
# 混合精度(automatic mixed precision, AMP): 让 PyTorch 自动挑精度
# matmul/conv 这类"耐低精度且吃 Tensor Core"的 → FP16
# softmax/layernorm/求和这类"怕溢出/怕精度损失"的 → 自动留 FP32
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    out = model(x_fp32)   # 注意输入还是 FP32! autocast 在算子内部按需转
```

> **为什么 autocast 是默认之选**:它把 §4 的"FP16 会在 softmax/求和处溢出"这个坑**自动帮你避开**了——敏感算子留 FP32,只在安全的矩阵乘上用 FP16。你几乎白拿了 FP16 的速度,又躲过了大部分溢出。BF16 场景同理换 `dtype=torch.bfloat16`。

## 8. 工业实践:常见陷阱与调试技巧

1. **用 `time.time()` 测 FP16 速度**:CUDA 异步(Day2/Day4/Day5 讲烂了),FP16 kernel 还没跑完 CPU 就返回了,测到的是假的。**必须 `torch.cuda.Event` + `synchronize()`**(§7.1 脚本就是范例)。

2. **只看逐元素误差、不看端到端**:§6 三把尺子的核心教训——**allclose 判死刑不判无罪**。你的 KV Cache/量化 PR,回归测试**必须带 PPL 或下游任务指标**,否则误差在"改答案"你都不知道。这是课题主线 4 验收的硬门槛。

3. **FP16 直接跑训练/含求和的推理,不加保护**:softmax 分母、LayerNorm 方差、loss 这些"大量求和"的地方 FP16 极易溢出/精度崩。**推理用 autocast(敏感算子自动留 FP32),训练用 BF16 或 FP16+loss scaling**。别手动全 `.half()` 上生产。

4. **INT8 用 per-tensor 量化后精度暴跌**:LLM 激活值有**离群值(outlier)**——绝大多数在 [-1,1],偶尔冒一个 100。用**整个张量一个 scale**,那个 100 会把 scale 撑大,导致正常值全挤在 INT8 的几个格子里,精度尽失。**解法:per-channel 量化(每列一个 scale)**,或 LLM.int8()/SmoothQuant 这类专门处理离群值的方案。这是量化 LLM 最经典的坑。

5. **以为"精度减半速度一定翻倍"**:只有 **memory-bound** 且**吃满 Tensor Core** 时才接近翻倍。小张量(launch-bound,Day4)、或 GPU 没有对应低精度单元时,收益大打折扣甚至变慢。**先判 bound(Day1 Roofline),再决定值不值得量化**。

6. **量化了权重却忘了 KV Cache**:decode 阶段 KV Cache 的搬运量随序列变长会**超过权重**。长序列场景,**量化 KV Cache(常用 FP16/INT8)比量化权重更能提速**——这正是知识点开头"KV Cache 常用 FP16"的实战理由。

7. **反量化位置错误**:INT8 权重必须在**参与计算前**反量化(或用 INT8 专用 kernel 直接算),不能量化了存下来、用的时候忘了 dequantize。数值会完全错乱且不报错。

## 9. 自测题(先合上笔记答,再翻对应节核对)

1. FP16 和 BF16 的位布局各是几位符号/指数/尾数?BF16 把省下的位挪去哪了,换来了什么、牺牲了什么?→ §2
2. **(核心)** 为什么"精度减半"能给 decode **直接**提速?为什么说在 memory-bound 场景"转换开销几乎免费"?→ §3
3. FP16 为什么会溢出(上限 65504 哪来的)?loss scaling 是怎么救下溢的梯度的?为什么训练偏 BF16?→ §4
4. 写出 INT8 量化/反量化公式。对称和非对称量化的 zero_point 分别是什么?各适合什么数据分布?→ §5
5. **(灵魂题)** "量化必须配误差度量"——三把尺子(allclose / logits cosine / PPL)各自查什么、各自的盲区是什么?为什么说 allclose"判死刑不判无罪"?→ §6
6. 本机实测:FP16 相对 FP32 加速约几倍?输出 cosine 约多少?若 cosine 掉得很低,最可能是什么原因、怎么救?→ §7
7. 把"decode memory-bound → 精度减半 → 搬运减半 → 提速 → 但引入误差 → 三把尺子验证 → 课题主线 4 误差可控可解释"连成一条因果链。→ §3 + §6

> 第 2、5 题是题眼。第 5 题能讲透"三把尺子的盲区",你就真正理解了课题主线 4 为什么把"误差可控可解释"单列为验收指标。

## 10. 与已有笔记 / 课题主线的串联

| 关联 | 关系 |
|---|---|
| [W7 Day1 · Roofline](./W7_Day1_Roofline_算术强度与H100脊点.md) | 精度减半 = 每字节不变但**总字节数减半** → 在 Roofline 上把 memory-bound 的墙**向右推**。判 bound 决定量化值不值得 |
| [W7 Day4 · 融合两笔账](./W7_Day4_KernelLaunch开销与算子融合_巨核动机.md) | 量化省的是**访存**(第二笔账)的另一条路:融合是"少搬几趟",量化是"每趟搬得更轻"。两者正交,可叠加 |
| [W7 Day5 · torch.compile](./W7_Day5_torch_compile_图优化自动版_巨核baseline.md) | torch.compile 也能和 AMP 叠加。量化是它之外**独立的一层** memory-bound 优化(知识点原话:融合是第一杀器,量化是第二) |
| W6 §5.3 · decode memory-bound([KV Cache 笔记](./人工智能(md)/W6_Day6_KV_Cache_两阶段_显存代价.md)) | 本篇 §3 的前提:正因为 decode 卡在搬运,精度减半才**直接**提速。没有 §5.3 这个结论,量化对 decode 就没有立足点 |
| W6 §4.4 · torch.equal vs allclose(同上笔记) | 本篇 §6 是它的**正式升级版**:从"优化版必须和朴素版数值等价"扩展到"量化引入误差时,用三把尺子量化并解释这个误差"。同一条职业素养血脉 |
| 小米课题**主线 4**(验证:数值一致/误差可控可解释) | **本篇是它的代码地基**:§6 三把尺子就是每个量化 PR 回归测试的模板。你交付物里"性能对比报告"的误差栏,靠的就是这套 |
| 小米课题主线 1(性能画像) | 量化前先用 Roofline/nsys 判 bound(Day1/Day2),量化后用同一套工具量提升——量化是"画像→优化→再画像"闭环里的一个动作 |
| [rnn_to_transformer_evolution](./rnn_to_transformer_evolution.md) | 那里讲 KV Cache 的显存代价;本篇补上"KV Cache 用 FP16 存能砍一半显存/带宽",是它的自然续集 |

## 11. 今日产出清单(对齐计划)

- [x] `precision_and_quantization.md`(本笔记正文,桌面按 `W7_Day6_*` 平铺;进仓时用计划里的文件名)
- [x] 位布局讲透:FP32/FP16/BF16 三者符号/指数/尾数 + **BF16 = FP32 砍掉后 16 位**的底层验证代码
- [x] 讲清"精度减半为何对 memory-bound 是大杀器"(接 W6 §5.3 + Day1 Roofline)
- [x] FP16 溢出/下溢 + loss scaling 底层机制 + "为何训练偏 BF16"
- [x] INT8 量化:scale/zero_point 公式、对称 vs 非对称、可跑的量化/反量化代码
- [x] **(最重要)** 数值一致性三把尺子(allclose / logits cosine / 端到端 PPL)+ 各自盲区 —— 课题主线 4"误差可控可解释"的代码落地
- [x] 配套脚本 `bench_precision.py`:nanoGPT-like block **FP32 vs FP16 速度差 + 数值误差**(本机 RTX 5060 可跑,不依赖 Triton)
- [ ] (本机今晚可做)真跑 `bench_precision.py`,把 FP16 加速倍数 + cosine 填进本笔记 §7.2,凑齐"速度/误差对比"实测数据
- [ ] (有 H100 / 真 nanoGPT 时)在真实 nanoGPT checkpoint 上加量化,用 §6.4 端到端 PPL 度量,作为 AMK/课题的量化基线

---

> **一句话收尾**:今天你把"量化能省显存"这句正确的废话,拆成了**可量化的收益(搬运量减半 → memory-bound 直接提速)+ 可量化的代价(三把尺子度量的数值误差)**。精度不是越高越好,而是"在够用的前提下越低越快"——而"够不够用",靠的正是 §6 那三把尺子。这就是课题主线 4"误差可控可解释"的全部内涵,也是你每写一个量化/巨核算子,都要回答的第二个问题(第一个是"快了多少",第二个是"错了多少、能不能解释")。
