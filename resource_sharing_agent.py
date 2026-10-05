"""A deterministic negotiating agent for sharing a constrained resource."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping


_EPSILON = 1e-9


@dataclass(frozen=True)
class ResourceClaim:
    """A participant's requested amount, reservation, and negotiation weight."""

    participant: str
    requested: float
    minimum: float = 0.0
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not self.participant.strip():
            raise ValueError("participant must not be empty")
        for label, value in (
            ("requested", self.requested),
            ("minimum", self.minimum),
            ("priority", self.priority),
        ):
            if not isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.requested < 0 or self.minimum < 0:
            raise ValueError("requested and minimum must be non-negative")
        if self.minimum > self.requested:
            raise ValueError("minimum cannot exceed requested")
        if self.priority <= 0:
            raise ValueError("priority must be greater than zero")


class Decision(str, Enum):
    ACCEPT = "accept"
    COUNTER = "counter"
    REJECT = "reject"


@dataclass(frozen=True)
class NegotiationResponse:
    decision: Decision
    reason: str
    counter_offer: Mapping[str, float] | None = None


class ResourceNegotiator:
    """Negotiate a feasible allocation using weighted fairness and concessions.

    ``round_number`` is zero-based. The agent starts by requesting its full
    amount and concedes toward its weighted fair allocation by ``max_rounds``.
    """

    def __init__(self, agent: str, claims: list[ResourceClaim]) -> None:
        self.agent = agent
        self.claims = tuple(claims)
        self._claims_by_name = {claim.participant: claim for claim in claims}
        if len(self._claims_by_name) != len(self.claims):
            raise ValueError("participant names must be unique")
        if agent not in self._claims_by_name:
            raise ValueError("agent must have a resource claim")

    def make_counter_offer(
        self,
        available: float,
        round_number: int,
        max_rounds: int = 5,
    ) -> dict[str, float]:
        """Create a feasible allocation, conceding toward weighted fairness."""
        self._validate_round(round_number, max_rounds)
        self._validate_available(available)

        fair = self._weighted_allocation(self.claims, available)
        own_claim = self._claims_by_name[self.agent]
        progress = round_number / max_rounds
        target = fair[self.agent] + (own_claim.requested - fair[self.agent]) * (1 - progress)
        own_allocation = min(available, max(target, min(own_claim.minimum, available)))

        offer = {self.agent: own_allocation}
        others = [claim for claim in self.claims if claim.participant != self.agent]
        remaining = max(0.0, available - own_allocation)
        offer.update(self._weighted_allocation(others, remaining))
        return offer

    def respond(
        self,
        proposal: Mapping[str, float],
        available: float,
        round_number: int,
        max_rounds: int = 5,
    ) -> NegotiationResponse:
        """Accept a fair-enough feasible proposal, otherwise counter or reject."""
        self._validate_round(round_number, max_rounds)
        self._validate_available(available)

        unknown = set(proposal) - self._claims_by_name.keys()
        if unknown:
            return NegotiationResponse(
                Decision.REJECT,
                f"unknown participant(s): {', '.join(sorted(unknown))}",
            )

        normalized: dict[str, float] = {}
        for claim in self.claims:
            amount = proposal.get(claim.participant, 0.0)
            if not isfinite(amount) or amount < 0 or amount > claim.requested:
                return NegotiationResponse(
                    Decision.REJECT,
                    f"invalid allocation for {claim.participant}",
                )
            normalized[claim.participant] = amount

        if sum(normalized.values()) > available + _EPSILON:
            return NegotiationResponse(Decision.REJECT, "proposal exceeds available resources")

        fair = self._weighted_allocation(self.claims, available)[self.agent]
        own_claim = self._claims_by_name[self.agent]
        progress = round_number / max_rounds
        threshold = fair + (own_claim.requested - fair) * (1 - progress)
        if normalized[self.agent] + _EPSILON >= threshold:
            return NegotiationResponse(Decision.ACCEPT, "proposal meets the current threshold")

        counter = self.make_counter_offer(available, round_number, max_rounds)
        return NegotiationResponse(
            Decision.COUNTER,
            "proposal is feasible but below the current threshold",
            counter,
        )

    @staticmethod
    def _validate_available(available: float) -> None:
        if not isfinite(available) or available < 0:
            raise ValueError("available must be finite and non-negative")

    @staticmethod
    def _validate_round(round_number: int, max_rounds: int) -> None:
        if max_rounds < 1 or not 0 <= round_number <= max_rounds:
            raise ValueError("round_number must be between zero and max_rounds")

    @staticmethod
    def _weighted_allocation(
        claims: tuple[ResourceClaim, ...] | list[ResourceClaim],
        available: float,
    ) -> dict[str, float]:
        allocations = {claim.participant: 0.0 for claim in claims}
        if not claims or available <= 0:
            return allocations

        minimum_total = sum(claim.minimum for claim in claims)
        if minimum_total > available:
            return ResourceNegotiator._weighted_fill(
                claims,
                available,
                {claim.participant: claim.minimum for claim in claims},
            )

        for claim in claims:
            allocations[claim.participant] = claim.minimum
        remaining = available - minimum_total
        extras = ResourceNegotiator._weighted_fill(
            claims,
            remaining,
            {claim.participant: claim.requested - claim.minimum for claim in claims},
        )
        return {
            participant: allocations[participant] + extras[participant]
            for participant in allocations
        }

    @staticmethod
    def _weighted_fill(
        claims: tuple[ResourceClaim, ...] | list[ResourceClaim],
        available: float,
        caps: Mapping[str, float],
    ) -> dict[str, float]:
        result = {claim.participant: 0.0 for claim in claims}
        active = {claim.participant: claim for claim in claims if caps[claim.participant] > 0}
        remaining = available

        while active and remaining > _EPSILON:
            total_weight = sum(claim.priority for claim in active.values())
            saturated = []
            for participant, claim in active.items():
                share = remaining * claim.priority / total_weight
                capacity = caps[participant] - result[participant]
                if share >= capacity - _EPSILON:
                    result[participant] += capacity
                    saturated.append(participant)

            if not saturated:
                for participant, claim in active.items():
                    result[participant] += remaining * claim.priority / total_weight
                break

            remaining = max(0.0, available - sum(result.values()))
            for participant in saturated:
                del active[participant]

        return result