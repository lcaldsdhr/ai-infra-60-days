"""Day 17-18: 两阶段 Pipeline Parallel（CPU / Gloo 可运行）。

目标：用真实 torch.distributed 点对点通信展示 PP 的数据流：
  - global/local batch=16，切为 num_microbatches=2；每个 microbatch 为 8 条样本
  - rank 0 持有 Stage 0，rank 1 持有 Stage 1
  - activation 从 rank 0 send 到 rank 1，activation gradient 反向 recv 回 rank 0
  - rank 0 的稳态调度为 F(i) -> B(i-1)，即 1F1B 的两 stage 版本

运行（仅需 PyTorch，无 GPU）：
  python code/day17-pp/two_stage_1f1b_cpu.py

可改参数：
  python code/day17-pp/two_stage_1f1b_cpu.py --batch-size 16 --num-microbatches 4
"""

import argparse
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


WORLD_SIZE = 2
INPUT_DIM = 6
HIDDEN_DIM = 8
NUM_CLASSES = 3


class Stage0(nn.Module):
    """第一段模型：真实项目中通常是一段连续 Transformer blocks。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class Stage1(nn.Module):
    """第二段模型：接收 activation 并计算 loss。"""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, activation):
        return self.classifier(activation)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def record(events, start, rank, action, microbatch):
    events.append((time.perf_counter() - start, rank, action, microbatch))


def gradient_norm(model):
    return sum(
        parameter.grad.detach().norm().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def stage0_1f1b(model, optimizer, inputs, microbatch_size, num_microbatches, events, start):
    """rank 0 的调度：warmup F(0)，稳态 F(i) + B(i-1)，最后 cooldown B(last)。"""
    saved_activations = []

    # Warmup：先让第一个 microbatch 进入流水线。
    activation = model(inputs[:microbatch_size])
    record(events, start, 0, "F", 0)
    saved_activations.append(activation)
    dist.send(activation.detach().contiguous(), dst=1)
    record(events, start, 0, "send activation", 0)

    # Steady state：计算当前 F，同时为上一份数据接收并消费 B。
    for microbatch in range(1, num_microbatches):
        begin = microbatch * microbatch_size
        end = begin + microbatch_size
        activation = model(inputs[begin:end])
        record(events, start, 0, "F", microbatch)
        saved_activations.append(activation)

        grad = torch.empty_like(saved_activations[microbatch - 1])
        dist.recv(grad, src=1)
        record(events, start, 0, "recv activation grad", microbatch - 1)
        saved_activations[microbatch - 1].backward(grad)
        record(events, start, 0, "B", microbatch - 1)

        dist.send(activation.detach().contiguous(), dst=1)
        record(events, start, 0, "send activation", microbatch)

    # Cooldown：最后一个 microbatch 的梯度回来后，完成最后一次反向。
    grad = torch.empty_like(saved_activations[-1])
    dist.recv(grad, src=1)
    record(events, start, 0, "recv activation grad", num_microbatches - 1)
    saved_activations[-1].backward(grad)
    record(events, start, 0, "B", num_microbatches - 1)
    grad_norm = gradient_norm(model)
    optimizer.step()
    return grad_norm


def stage1_1f1b(model, optimizer, labels, microbatch_size, num_microbatches, events, start):
    """rank 1：每收到一个 activation，完成对应 F/B 并把边界梯度送回 rank 0。"""
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0

    for microbatch in range(num_microbatches):
        begin = microbatch * microbatch_size
        end = begin + microbatch_size
        activation = torch.empty(microbatch_size, HIDDEN_DIM)
        dist.recv(activation, src=0)
        record(events, start, 1, "recv activation", microbatch)
        activation.requires_grad_(True)  # 边界 activation 是本 stage 反向的 leaf。

        logits = model(activation)
        # 除以 microbatch 数，保证累积梯度等价于一个 batch_size=16 的平均 loss。
        loss = criterion(logits, labels[begin:end]) / num_microbatches
        loss_sum += loss.item()
        record(events, start, 1, "F + loss", microbatch)
        loss.backward()
        record(events, start, 1, "B", microbatch)

        dist.send(activation.grad.contiguous(), dst=0)
        record(events, start, 1, "send activation grad", microbatch)

    grad_norm = gradient_norm(model)
    optimizer.step()
    return loss_sum, grad_norm


def worker(rank, batch_size, num_microbatches, port):
    # 显式指定 loopback，避免 Windows/Gloo 根据主机名选择不可用网卡。
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    torch.manual_seed(20260716)  # 两个 rank 对数据和初始化使用同一确定性种子。

    microbatch_size = batch_size // num_microbatches
    inputs = torch.randn(batch_size, INPUT_DIM)
    labels = torch.randint(NUM_CLASSES, (batch_size,))
    model = Stage0() if rank == 0 else Stage1()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)

    # 统一时间起点，收集后的事件顺序才可作为时间线阅读。
    dist.barrier()
    events = []
    start = time.perf_counter()
    if rank == 0:
        stats = {"loss": None, "grad_norm": stage0_1f1b(
            model, optimizer, inputs, microbatch_size, num_microbatches, events, start
        )}
    else:
        loss_sum, grad_norm = stage1_1f1b(
            model, optimizer, labels, microbatch_size, num_microbatches, events, start
        )
        stats = {"loss": loss_sum, "grad_norm": grad_norm}

    gathered_events = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_events, events)
    gathered_stats = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_stats, stats)
    if rank == 0:
        print(f"local_batch_size={batch_size}, num_microbatches={num_microbatches}, "
              f"microbatch_size={microbatch_size}")
        print(f"mean loss before optimizer.step = {gathered_stats[1]['loss']:.4f}")
        print("\n时间线（同一行表示一个 rank 上按时间发生的事件）：")
        for _, event_rank, action, microbatch in sorted(sum(gathered_events, [])):
            print(f"rank {event_rank}: {action:<22} mb{microbatch}")
        norms = [item["grad_norm"] for item in gathered_stats]
        if not all(norm > 0 for norm in norms):
            raise RuntimeError(f"梯度校验失败：{norms}")
        print(f"\n验证通过：两个 stage 的梯度范数分别为 {norms[0]:.4f}, {norms[1]:.4f}，"
              "均完成反向传播并各自执行了一次 optimizer.step。")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-microbatches", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_microbatches <= 0:
        raise ValueError("batch-size 和 num-microbatches 必须为正数")
    if args.batch_size % args.num_microbatches:
        raise ValueError("batch-size 必须能被 num-microbatches 整除")

    mp.spawn(worker, args=(args.batch_size, args.num_microbatches, free_port()), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
