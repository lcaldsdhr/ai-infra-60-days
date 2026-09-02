# Toy KV Cache Attention

这个纯 Python 单头注意力实验比较两种 Decode：

- 无缓存：每一步对完整历史重新计算 K/V；
- 有缓存：Prompt 的 K/V 只算一次，每一步只追加新 token 的 K/V。

两条路径使用相同 Q/K/V 矩阵，并断言每步 attention 输出相同。

## 运行

```powershell
python RL_go/code/toy_kv_cache_attention/demo.py
```

预期关键输出：

```text
K/V projections without cache: 30
K/V projections with cache:    12
saved projections:              18

Conclusion: outputs match; the cache removes repeated historical K/V projections.
```

## 读代码时关注

1. `full_recompute()` 每步重建所有历史 `keys/values`；
2. `cached_decode()` 先 Prefill，再为每个新 token 追加一组 K/V；
3. `query` 仍与全部历史 keys 匹配，因此 KV Cache 不会让长上下文读取免费。

## 限制

这不是 Transformer：没有多头、causal mask、RoPE、MLP、残差、批处理和 GPU kernel。它只隔离“缓存 K/V 是否改变结果、减少哪部分重复计算”这一件事。
