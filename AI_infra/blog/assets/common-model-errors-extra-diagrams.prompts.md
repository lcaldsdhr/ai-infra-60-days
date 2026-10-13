# 常见报错博客：新增三张示意图的提示词

生成日期：2026-10-13。方式：内置 imagegen；默认技术图风格。最终图片已视觉检查。图示不是实测数据或特定故障诊断；未运行 GPU 实验。

## OOM 显存账本

最终资产：`oom-memory-ledger-and-fixes.png`。

### oom_prompt

Create a high-resolution wide 16:9 Chinese educational technical infographic titled "OOM：显存花在哪里？". White engineering canvas, thin cobalt-blue rounded panels, large centered cobalt title, dark navy legible Simplified Chinese type, flat pale-blue/teal/amber/coral semantic blocks, numbered stages and pale-blue footer. No 3D, ornament, invented measurements.
Three panels read left to right.
① "训练显存账本" show stacked named rectangles representing conceptual components, NOT proportional sizes: "模型参数", "梯度", "优化器状态", "保存的激活", "临时空间 / 通信 buffer". Show forward -> backward -> optimizer timeline with activation high during forward/backward and optimizer state may appear at first step. Concise captions "长序列增大激活" and "首次 step 可能创建优化器状态".
② "推理显存账本" show "模型权重", "KV Cache", "临时空间 / 图捕获". Next to KV rectangular token blocks show prompts A,B,C growing over decode time; labels "并发 × 序列长度 → KV 压力", "多次 rollout 增加生成负载". Precise note "缓存共享与模型架构会改变实际占用".
③ "用证据选修复" three clean rows "激活峰值 → microbatch / 重计算", "训练状态 → 分片 / offload", "KV 容量 → 限并发 / 长度预算". Below three non-proportional nested boxes: outer "进程总显存", inner "分配器 reserved", inner subset "活跃 tensor allocated"; outside reserved but inside process box label "context / 外部库". Caption "reserved 高不自动等于泄漏", "empty_cache 不释放活跃 tensor".
Footer "记录阶段峰值 → 找主要占用者 → 修复 → 原工作负载复测". Small note "示意账本；色块不代表实测比例". Chinese labels exact readable no long paragraphs.

## 异步报错与通信超时

最终资产：`cuda-async-and-nccl-error-cascade.png`。

### async_prompt

Use case infographic-diagram. Create wide 16:9 white Chinese technical diagram title "为什么最后的报错不是最早的故障？". Clean cobalt-blue outlines and large title, dark navy Chinese labels, flat blue normal/amber waiting/coral failure/teal diagnostic colors, crisp readable typography. Two numbered panels side by side, footer flow. Conceptual examples only, no actual times, no fabricated logs.
① "CUDA 异步：提交位置与发现位置" swimlanes "CPU" and "GPU", time flows right. CPU lane cells "提交算子 A" -> "提交算子 B" -> "同步 / 读取结果" coral report "此处报告错误". GPU lane cells "执行 A" -> coral "B 非法访问" -> "错误状态". Dashed vertical connector from CPU submits B to GPU execution B, coral dashed arrow from GPU B to CPU later report. Labels "错误发生在 B", "traceback 可能指向后续同步", "调试：CUDA_LAUNCH_BLOCKING=1". Make no implication that all later kernels execute after illegal access.
② "分布式：一个 rank 失败，其他 rank 等待" 3 horizontal swimlanes Rank 0, Rank 1, Rank 2 time flows right. All first have "计算 step". Rank 2 has coral "错误标签 → GPU 断言" then "未进入 AllReduce". Ranks 0 and 1 each amber "进入 AllReduce → 等待" and at right coral "NCCL timeout". Bracket between ranks 0/1 waiting and rank2 absent labelled "collective 缺席". Beneath teal "先查各 rank 最早错误" and "再核对通信顺序 / shape / group". Small precise note "网络与资源问题也可导致 timeout；此图只示范一种因果链".
Footer numbered compact pipeline "完整日志 → 最早异常 → 最小复现 → 同步 / 通信诊断 → 修复后复测". Lower small note "同步调试会改变性能与时序，复测时恢复配置". Render requested text exactly and keep spacing ample.

### async_edit_prompt

Edit this technical infographic minimally. Keep all layout, arrows, diagrams, colors and Chinese text unchanged. In the two red timeout boxes at the right of Rank 0 and Rank 1, replace the existing misspelled English labels with exactly 'NCCL timeout' in both boxes. Ensure spelling letter by letter N C C L, then space, t i m e o u t. Do not change other elements.

### async_edit2_prompt

Edit only the two red error boxes in the right-hand panel for Rank 0 and Rank 1. Remove all current English text inside both boxes and replace it with Chinese text exactly '通信超时'. Keep everything else including title, all arrows, timelines and other labels unchanged.

## verl 样本与 token 对齐

最终资产：`verl-sample-token-alignment-errors.png`。

### alignment_prompt

Create wide 16:9 Chinese technical educational infographic title "verl：shape 一致，为什么仍可能训练错？". White engineering canvas, cobalt blue thin rounded panels, numbered headings, navy clear text, flat blue/teal/amber/coral colors. Precisely readable labels, uncluttered.
Three numbered panels left to right.
① "样本维：一行是一次 rollout" table exactly columns "索引", "uid", "回答", "reward"; rows "0 | A | A1 | 1", "1 | A | A2 | 0", "2 | B | B1 | 0", "3 | B | B2 | 1". Below two short callouts "索引 2 → B1 一次 rollout", "uid=B → B1、B2 同题一组".
② "错配：长度相同，身份不同" show original responses column A1,A2,B1,B2 and correct advantages column adv(A1),adv(A2),adv(B1),adv(B2) aligned in teal. Below show a swapped advantages column with adv(B2) assigned A1 and adv(A1) assigned B2, those two rows coral; other rows matching. Need simple arrows illustrate same index mismatched identity, not memory address error. Labels "只重排 responses，未同步其他字段", "shape 仍合法，梯度却错误". Below "修复：所有样本字段使用同一索引映射".
③ "token 维：同一动作位置" a short context strip "prompt | y0 | y1 | pad" colored prompt gray, y0,y1 blue, pad gray. Under separate only-response aligned table columns "y0","y1","pad"; row names "log-prob","advantage","mask" with values "logp(y0),logp(y1),—"; "a0,a1,—"; "1,1,0". Emphasize "context 预测下一个 token；核对 label shift". Additional concise callouts "old / ref / rollout：概率来源不同", "工具 observation 是上下文，动作 mask 按目标定义".
Bottom pale-blue footer "样本身份对齐 + token 位置对齐 + 策略版本可追溯 → 训练信号可信". Tiny note "教学示例；不是内存踩踏；真实字段与 mask 依版本和训练目标核对". No made-up performance metrics.

### alignment_edit_prompt

Make a minimal text edit to this Chinese infographic. In the middle panel, the red heading above the right-hand mismatched table currently reads '只重排 responses，未同步其他字段'. Replace that heading with exactly '部分字段重排，未同步其他字段'. The table represents swapped advantages and unchanged responses, so this general heading matches it. Keep every other table entry, text, diagram, arrow, color and layout unchanged.
