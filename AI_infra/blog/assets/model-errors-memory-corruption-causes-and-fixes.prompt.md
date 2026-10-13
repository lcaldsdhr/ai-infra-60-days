# 模型报错与内存踩踏：生成说明

- 生成方式：内置 image_gen，2026-10-13。
- 图片：`model-errors-memory-corruption-causes-and-fixes.png`。
- 类型：中文技术教学示意图，不是一次真实故障的诊断结果。
- 风格：白底、钴蓝描边、编号分区、蓝/青/黄/珊瑚语义配色、底部排查流程条。
- 内容检查：已视觉检查文字、布局与箭头；未执行图中的调试命令。
- 工具事实参考：[NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。memcheck 检查越界及非对齐访问；racecheck 主要检查 shared memory 数据访问竞态，不能替代全部并发问题的分析。

## 最终提示词

Use case: infographic-diagram. Create a polished wide 16:9 high-resolution Chinese technical teaching infographic titled "模型报错与内存踩踏：现象、原因、修复". White background, thin cobalt blue rounded panels, numbered sections, dark navy crisp readable Simplified Chinese typography, pale blue valid memory, teal fixes, amber suspect reads/timing, coral corrupting writes. No 3D/no decorative AI imagery. Prioritize clear concise text and mechanism diagrams.
Top horizontal section ① "上层现象：报错不等于根因" shows three small labels "illegal memory access", "段错误 / access violation", "随机崩溃 / 输出异常". Beneath in Chinese: "CUDA 异步执行：报错位置可能晚于出错算子". A dashed warning secondary label "NCCL 超时：可能是某个 rank 先失败的后果". Symptoms are possible clues, not proof or one-to-one diagnoses.
Middle large section ② "底层原因与对应修复" four equally sized cards in 2x2, each contains a tiny memory/timeline mechanism drawing and short cause + fix:
A "越界写入" diagram buffer A valid cells [0..3], stray write A[4] crosses boundary into neighboring buffer B shown coral. "原因：索引越界 / stride 或偏移算错" "修复：边界 mask；核对 shape / stride" "检测：Compute Sanitizer memcheck"
B "释放后使用" diagram allocate → free → stale pointer writes reused memory. "原因：异步任务未结束，内存已释放或复用" "修复：管理生命周期；等待 event / stream" "检测：memcheck；CPU 用 AddressSanitizer"
C "并发写入竞态" diagram two threads both write same valid address, no out of bounds; wrong value coral. "原因：缺少同步或原子操作" "修复：明确所有权；同步或 atomic" "检测：racecheck 检查 shared memory 竞态" small precise note "竞态可破坏数据，不必越界"
D "原生算子接口错误" diagram model tensor → C++ / CUDA kernel → wrong address from wrong dtype/stride/size contract. "原因：dtype / 布局假设错误；长度计算溢出" "修复：校验接口；修正寻址；替换缺陷算子" "检测：最小复现 + memcheck / ASan"
Third narrow section ③ "容易混淆的其他错误" three side-by-side compact blocks: "OOM：容量不足，通常不是踩踏", "device-side assert：常见索引越界，不能单凭它认定踩踏", "DataProto 样本错配：训练信号错误，通常不是踩踏". Explicit small definition "内存踩踏通常指非法写入破坏其他内存；非法读取也会报错，但不一定踩踏。"
Bottom pale-blue footer section ④ "排查路径" with clean arrows: "最早错误 → 单进程最小复现 → 同步定位 → 内存检测 → 修复后复测". Small label "CUDA_LAUNCH_BLOCKING=1 仅用于调试定位". Small note "示意图；未针对某次真实故障作诊断". Keep all text readable and faithfully spelled.
