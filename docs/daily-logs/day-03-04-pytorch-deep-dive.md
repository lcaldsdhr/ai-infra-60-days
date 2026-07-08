# Day 3-4 笔记：PyTorch 深度

> **目标**：深入 PyTorch 内部机制——autograd 计算图、nn.Module 三件套、AMP 混合精度、torch.compile 入门。

---

## 一、autograd 原理

### 1. 计算图构建规则

```python
a = torch.tensor([2.0], requires_grad=True)   # 在图上（leaf）
b = torch.tensor([3.0])                        # 不在图上（requires_grad=False）
c = torch.tensor([4.0], requires_grad=False)   # 不在图上

# 规则：只要所有输入不需要梯度，输出也不需要梯度
r = b * 2   # r.requires_grad = False  ← 全不需要，跳过
s = a * 3   # s.requires_grad = True   ← 有一个需要，全追踪
```

每次 forward 在背后建一个有向无环图（DAG）：
- 节点：操作（MulBackward0, PowBackward0 等）
- 边：数据流

每个 tensor 有两个关键属性：

| 属性 | 含义 |
|---|---|
| `.grad_fn` | 指向生成这个 tensor 的反向函数。leaf tensor（用户直接创建）的 grad_fn=None |
| `.is_leaf` | 是否为图的叶子节点 |

### 2. 反向传播时序

`z.backward()` 内部流程：
1. 拓扑排序整个计算图
2. 从输出往输入反向遍历，每个节点调 grad_fn 算局部梯度
3. 链式法则累加梯度到 leaf 节点的 `.grad`

```python
x = tensor([1.0])      # leaf, grad_fn=None
y = x * 2               # non-leaf, grad_fn=MulBackward0
z = y ** 2              # non-leaf, grad_fn=PowBackward0
z.backward()
print(x.grad)  # tensor([8.])
# 链式法则：dz/dx = dz/dy · dy/dx = 2y · 2 = 4y = 8x = 8

print(y.grad)  # None（non-leaf 用完即释放）
y.retain_grad()         # 告诉 autograd 保留 y 的梯度
z.backward()
print(y.grad)  # tensor([4.])
```

### 3. leaf vs non-leaf

| 属性 | Leaf | Non-leaf |
|---|---|---|
| 谁创建的 | 用户直接创建（或 detach）| 运算产生 |
| `grad_fn` | None | 指向反向函数 |
| `is_leaf` | True | False |
| 反向后 `.grad` | ✅ 保留梯度 | ❌ 不保留（省显存）|

**为什么 non-leaf 不保留梯度**：中间节点的梯度只在反向传播过程中使用一次，传播到上游后就没用了，PyTorch 立即释放以节省显存。

### 4. 非标量 backward

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward()   # ❌ RuntimeError

# 正确做法：传 gradient 参数（形状与 y 一致）
y.backward(torch.tensor([1.0, 1.0, 1.0]))
# 等价于：(y * torch.tensor([1.0, 1.0, 1.0])).sum().backward()

# 更常见的做法是：
loss = y.sum()     # 把非标量变成标量
loss.backward()    # ✅
```

**为什么报错**：梯度定义在标量上，向量对向量求导得到的是 Jacobian 矩阵，autograd 不知道你想要哪个方向的总梯度。传入 gradient 参数等价于指定 v^T · J（向量-Jacobian 积）中的 v。

---

## 二、nn.Module / Parameter / Buffer

| 条目 | 类 | requires_grad | 是否被 optimizer 看到 | 是否在 state_dict 中 |
|---|---|---|---|---|
| 可训练权重 | nn.Parameter | 默认 True | ✅ | ✅ |
| 运行缓存 | register_buffer() | 自定义 | ❌ | ✅ |
| 普通属性 | 普通 Tensor | 自定义 | ❌ | ❌ |

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(10))         # ✅ 被优化，被保存
        self.register_buffer("running_mean", torch.zeros(10))  # ❌ 不被优化，但被保存
        self.temp = torch.randn(10)                     # ❌ 既不优化也不保存
```

**面试题**：model.state_dict() 包含哪几个？

> 答案：self.w（Parameter）和 self.running_mean（buffer）都会出现，self.temp 不会。决定因素是 nn.Module 的 _parameters 和 _buffers 两个 OrderedDict。

---

## 三、AMP 混合精度

### 1. 位级结构

```
     S 指数(E)  尾数(M)
FP32: 1 + 8 + 23   → 范围 ±3.4e38，精度 7 位小数
BF16: 1 + 8 + 7    → 范围 ±3.4e38（同 FP32），精度 2-3 位小数
FP16: 1 + 5 + 10   → 范围 ±65504，精度 3-4 位小数
```

| 特性 | FP16 | BF16 |
|---|---|---|
| 指数位 | 5 → 最大 65504 | 8 → 最大 3.4e38（同 FP32）|
| 下溢风险 | ❌ 梯度 < 2⁻²⁴ 变 0 | ✅ 范围够用 |
| loss scaling | ✅ 需要 | ❌ 不需要 |
| 精度损失 | 尾数 10 位 | 尾数 7 位（精度比 FP16 差）|
| 硬件支持 | 所有 GPU | A100+ / Ascend NPU |

**BF16 不需要 loss scaling 的原因**：指数位 8 位，范围与 FP32 完全一致，能容纳所有梯度值。FP16 指数位 5 位，训练时梯度值通常在 2⁻³⁰~2⁻¹⁰ 级，小于 2⁻²⁴ 的梯度下溢为 0。

### 2. GradScaler 原理

```python
scaler = GradScaler()

# 前向用 autocast 自动选 FP16 算子
with autocast(dtype=torch.float16):
    out = model(**batch)
    loss = loss_fn(out, batch["labels"])

# 反向：loss 被 scale 防止梯度下溢
scaler.scale(loss).backward()     # loss × scale → backward
scaler.unscale_(optim)            # 可选，但 clip 前要调
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optim)                # 内部 ÷scale 还原梯度 → 权重更新
scaler.update()                   # 检测 NaN → 调整 scale
```

**scaler.step 内部逻辑**：
- 检测到 NaN/Inf → 跳过更新 → scale × 0.5
- 连续 2000 步无 NaN → scale × 2
- 正常 → ÷scale 还原梯度 → 权重更新

### 3. NaN/Inf 排查 6 步

1. 检查数据（标签是否越界，输入是否有 NaN）
2. 降低 learning rate（大 lr → 梯度爆炸 → NaN）
3. 开启 gradient clipping（clip_grad_norm_(1.0)）
4. 换 BF16（FP16 可能下溢）
5. GradScaler 的初始 scale 从 2¹⁶ 降到 2⁸ 或 2⁴
6. `torch.autograd.set_detect_anomaly(True)` 定位第一个产生 NaN 的算子

---

## 四、torch.compile 入门

**一句话**：把 Python 代码编译成高效的 Triton kernel，减少 Python 解释器开销和 kernel launch 次数。

**内部 3 步**：
1. 抓取 FX graph（前向计算图）
2. 图优化（算子融合、死代码消除）
3. Inductor 后端生成 Triton/CPU 代码

```python
model = torch.compile(model, mode="reduce-overhead")   # 推理用
model = torch.compile(model, mode="max-autotune")       # 训练用
```

**代价**：首次编译慢（几十秒），调试不友好，动态 shape 会重新编译。

---

## 五、训练 8 步循环

### 标准顺序

```
① batch.to(device)     数据搬 GPU
② optim.zero_grad()    清空上一步梯度
③ out = model(**batch) 前向，算 loss
④ loss.backward()      反向，累加梯度
⑤ clip_grad_norm_()    梯度裁剪
⑥ optim.step()         优化器更新权重
⑦ sched.step()         学习率调度器步进
⑧ ema.update()         EMA 影子权重更新
```

### 记忆口诀

> 搬 清 前 反 裁 更 调 影
> 准备段：搬 清  →  计算段：前 反 裁  →  更新段：更 调 影

---

## 六、面试高频题汇总

| 题 | 一句话答 |
|---|---|
| `model.eval()` vs `torch.no_grad()`? | eval() 关 Dropout/BN，no_grad() 不建计算图。都必须在 eval 时用 |
| Parameter vs Buffer? | Parameter 被 optimizer 看到且保存，Buffer 只保存不优化 |
| BF16 为什么不需要 loss scaling? | 指数位 8 位同 FP32，范围够大 |
| FP16 NaN 怎么排查? | 数据 → lr → clip → BF16 → scaler → anomaly detect |
| torch.compile 什么原理? | 图编译 + 算子融合 + Triton 生成 |
| 非标量 backward 为什么报错? | 不知道要哪个方向的梯度，需传 gradient 参数 |
| **总结题**：8 步顺序要倒背如流——准备段起手，计算段发力，更新段收尾 |
