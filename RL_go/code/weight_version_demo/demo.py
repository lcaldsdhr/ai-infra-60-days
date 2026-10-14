"""CPU-only teaching model of weight synchronization, not an RL trainer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    version: int
    weights: tuple[float, float]


@dataclass(frozen=True)
class Experience:
    source_version: int
    source_weights: tuple[float, float]


class RolloutEngine:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.reported_version = policy.version
        self.accepting_requests = True

    def generate(self) -> Experience:
        if not self.accepting_requests:
            raise RuntimeError("generation paused during sync")
        # No token generation here: capture the actual policy used for a request.
        return Experience(self.reported_version, self.policy.weights)

    def sync(self, actor: Policy) -> None:
        self.accepting_requests = False
        try:
            # Immutable assignment represents a completed weight application.
            # Real distributed transfer/application needs backend coordination.
            self.policy = Policy(actor.version, tuple(actor.weights))
            self.reported_version = actor.version
        finally:
            self.accepting_requests = True


def check_synced(actor: Policy, engines: list[RolloutEngine]) -> None:
    for engine in engines:
        if engine.reported_version != actor.version:
            raise ValueError("version mismatch")
        if engine.policy.weights != actor.weights:
            raise ValueError("same label, different weights")


def expect_rejected(actor: Policy, engines: list[RolloutEngine], label: str) -> None:
    try:
        check_synced(actor, engines)
    except ValueError as error:
        print(f"{label}: rejected ({error})")
    else:
        raise AssertionError(f"{label} unexpectedly passed")


def main() -> None:
    actor = Policy(17, (0.6, 0.4))
    engines = [RolloutEngine(actor), RolloutEngine(actor)]
    check_synced(actor, engines)
    history = engines[0].generate()
    print(f"history: version={history.source_version}, weights={history.source_weights}")

    # Simulate an Actor update; this is not gradient descent.
    actor = Policy(18, (0.75, 0.25))
    expect_rejected(actor, engines, "actor updated, rollout stale")

    for engine in engines:
        engine.reported_version = 18
    expect_rejected(actor, engines, "only changed labels")

    engines[0].sync(actor)
    expect_rejected(actor, engines, "only synced replica 0")

    engines[1].sync(actor)
    check_synced(actor, engines)
    current = engines[1].generate()
    assert current.source_version == 18
    assert current.source_weights == actor.weights
    assert history.source_version == 17
    assert history.source_weights == (0.6, 0.4)
    print(f"new request: version={current.source_version}, weights={current.source_weights}")
    print("PASS: all replicas synced; historical provenance unchanged")


if __name__ == "__main__":
    main()
