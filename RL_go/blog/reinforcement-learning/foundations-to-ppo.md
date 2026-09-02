# 从 MDP、回报与优势到 PPO

## 1. 强化学习到底在优化什么

监督学习告诉模型每个输入的正确答案；强化学习只定义交互规则和奖励，让策略通过采样行为提高期望回报。

```text
state s_t ──► policy πθ(a|s) ──► action a_t
    ▲                                  │
    └──── environment: r_t, s_{t+1} ◄──┘
```

MDP 通常写成 `(S, A, P, R, γ)`：状态、动作、转移概率、奖励和折扣因子。策略目标是最大化期望折扣回报：

\[
G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots
\]

## 2. Value、Q 与 Advantage

| 量 | 含义 | 直觉 |
| --- | --- | --- |
| \(V^\pi(s)\) | 从状态 `s` 按策略继续的期望回报 | 当前局面的平均前景 |
| \(Q^\pi(s,a)\) | 在 `s` 先做 `a` 再按策略继续的期望回报 | 这个动作之后的前景 |
| \(A^\pi(s,a)=Q-V\) | 动作相对当前平均水平的增益 | 这个动作比“平时表现”好多少 |

Advantage 为正，就提高该动作概率；为负，就降低。baseline 不改变期望梯度方向，却能显著降低方差。GRPO 的组均值也是 baseline 思想的一种具体实现。

## 3. REINFORCE：最直接的策略梯度

对采样轨迹，REINFORCE 使用：

\[
\nabla_\theta J(\theta)\approx\sum_t G_t\nabla_\theta\log\pi_\theta(a_t|s_t)
\]

若一条轨迹最终得高分，就整体提高其中动作的 log-prob；低分则相反。优点是简单、无须学习环境模型，缺点是整条轨迹回报噪声大，长轨迹信用分配困难。

## 4. Actor-Critic 与 GAE

Actor 产生动作，Critic 学习 `V(s)`。一步 TD residual：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

GAE 将多个长度的 TD residual 加权：

\[
\hat A_t^{GAE}=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
\]

`λ` 在一步 TD 的低方差与 Monte Carlo 的低偏差之间折中。实现最容易错的是终止 mask、padding mask 与 bootstrap：真正 terminal 后不能继续使用下一个状态的 Value。

## 5. PPO 为什么需要 ratio 与 clip

rollout 由旧策略 \(\pi_{old}\) 采样，训练时参数已经变为 \(\pi_\theta\)。重要性比率：

\[
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
=\exp(\log\pi_\theta-\log\pi_{old})
\]

PPO clipped objective：

\[
L^{CLIP}=\mathbb E[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]
\]

它不是简单把所有 ratio 裁掉，而是阻止“已经朝有利方向走得太远”继续获得优化收益：

| Advantage | ratio 极端方向 | Clip 的作用 |
| --- | --- | --- |
| `A > 0` | 新策略把动作概率抬得过高 | 上界限制继续奖励 |
| `A < 0` | 新策略把动作概率压得过低 | 下界限制过度惩罚 |

运行[最小 PPO Clip 实验](../../code/ppo_clip_demo/README.md)，可直接看到 `ratio=1.5, A=+1` 被限制到 `1.2`，以及 `ratio=0.5, A=-1` 的目标被限制到 `-0.8`。

## 6. 训练稳定性不能只看 loss

PPO/LLM RL 至少要联看：reward、entropy、KL、clip fraction、advantage 均值/方差、response length、Value loss（若有）、验证集质量。常见误判包括：

- reward 上升但 verifier 被钻漏洞；
- loss 很平稳但大量组全对/全错、实际没有梯度；
- entropy 快速下降，模型变得确定但泛化下降；
- KL/ratio 异常来自训推 log-prob 不一致，而非算法超参数。

## 7. 一手入口

- [Sutton & Barto《Reinforcement Learning》](http://incompleteideas.net/book/RLbook2020.pdf)
- [Policy Gradient Theorem](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- [GAE](https://arxiv.org/abs/1506.02438)
- [PPO](https://arxiv.org/abs/1707.06347)
- [完整一手资料索引](../../docs/research/learning-curriculum-primary-sources.md)
