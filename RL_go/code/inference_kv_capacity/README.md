# KV Cache 容量计算器

这个标准库实验把 KV Cache 容量公式变成可修改参数，帮助理解层数、KV heads、head dim、精度、上下文和并发如何共同决定显存。

## 运行

```powershell
python RL_go/code/inference_kv_capacity/demo.py
```

默认参数：28 层、4 个 KV heads、head dim 128、BF16/FP16（2 bytes）、8192 token、64 并发、40 GiB KV 预算。

预期关键输出：

```text
per cached token:       56.000 KiB
per 8192-token sequence: 448.000 MiB
64 full sequences:    28.000 GiB
max full sequences in 40.0 GiB KV budget: 91
```

自定义示例：

```powershell
python RL_go/code/inference_kv_capacity/demo.py `
  --layers 32 --kv-heads 8 --head-dim 128 `
  --context 16384 --concurrency 32 --dtype-bytes 2
```

## 边界

公式只估计 decoder KV，不包含模型权重、activation、CUDA workspace、allocator 元数据、分页未满块和运行时保留显存。它用于第一性容量判断，不替代真实引擎 profiling。
