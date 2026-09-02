# RL / verl 代码实验

本目录只存放可运行、可复现的实验代码；一个实验一个子目录。

每个实验至少包含：

- `README.md`：依赖、运行命令、预期输出和已知限制；
- 源码与最小配置；
- 不含权重、数据集、checkpoint 和凭据。

涉及上游源码时，记录它对应的 `RL_go/verl` gitlink 提交。

## 当前实验

| 实验 | 内容 | 运行方式 |
| --- | --- | --- |
| [toy_kv_cache_attention](toy_kv_cache_attention/README.md) | 用单头注意力验证缓存/重算结果一致，并统计 K/V 重复投影 | `python RL_go/code/toy_kv_cache_attention/demo.py` |
| [inference_kv_capacity](inference_kv_capacity/README.md) | 估算每 token、单序列和并发请求的 KV Cache 容量 | `python RL_go/code/inference_kv_capacity/demo.py` |
| [ppo_clip_demo](ppo_clip_demo/README.md) | 手算正负优势下的 importance ratio 与 PPO clipped objective | `python RL_go/code/ppo_clip_demo/demo.py` |
| [grpo_advantage_demo](grpo_advantage_demo/README.md) | 组内优势、token 广播与一次简化策略更新 | `python RL_go/code/grpo_advantage_demo/demo.py` |
