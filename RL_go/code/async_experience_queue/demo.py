"""Teaching subset of verl MessageQueue semantics; no Ray or model dependency."""

import asyncio
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Experience:
    name: str
    source_version: int
    response: str


class MiniQueue:
    def __init__(self, capacity: int = 2):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items = deque()
        self.condition = asyncio.Condition()

    async def put_sample(self, sample: Experience) -> bool:
        async with self.condition:
            dropped = len(self.items) >= self.capacity
            if dropped:
                self.items.popleft()
            self.items.append(sample)
            self.condition.notify_all()
            return not dropped

    async def get_sample(self) -> tuple[Experience, int]:
        async with self.condition:
            while not self.items:
                await self.condition.wait()
            sample = self.items.popleft()
            return sample, len(self.items)


async def main() -> None:
    queue = MiniQueue()
    await queue.put_sample(Experience("A", 17, "answer A"))
    await queue.put_sample(Experience("B", 17, "answer B"))
    # Changing a separate engine's version does not mutate queued records.
    loaded_version = 18
    first, remaining = await queue.get_sample()
    assert first.source_version == 17 and remaining == 1
    print(f"rollout now v{loaded_version}; consumed {first.name} from v{first.source_version}")

    queue = MiniQueue()
    await queue.put_sample(Experience("A", 17, "answer A"))
    await queue.put_sample(Experience("B", 17, "answer B"))
    result = await queue.put_sample(Experience("C", 18, "answer C"))
    second, _ = await queue.get_sample()
    third, _ = await queue.get_sample()
    assert result is False and (second.name, third.name) == ("B", "C")
    print(f"full queue put C returned {result}; retained [{second.name}, {third.name}]")

    empty = MiniQueue()
    consumer = asyncio.create_task(empty.get_sample())
    await asyncio.sleep(0)  # Yield once; no wall-clock delay is introduced.
    assert not consumer.done()
    await empty.put_sample(Experience("D", 18, "answer D"))
    received, remaining = await consumer
    assert received.name == "D" and remaining == 0
    print(f"waiting consumer resumed with {received.name}; remaining={remaining}")
    print("PASS: provenance, oldest eviction, and asynchronous waiting")


if __name__ == "__main__":
    asyncio.run(main())
