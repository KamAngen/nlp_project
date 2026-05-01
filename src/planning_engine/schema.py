from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolPlanStep:
    tool_name: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None
    depends_on: list[int] = field(default_factory=list)
    is_optional: bool = False
    fallback_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionPlan:
    intent: str
    objective: str
    steps: list[ToolPlanStep] = field(default_factory=list)
    response_style: str = "structured"
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_user_input: bool = False
    user_input_prompt: str | None = None
    max_steps: int = 10
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    def add_step(
        self,
        tool_name: str,
        reason: str,
        arguments: dict[str, Any] | None = None,
        *,
        condition: str | None = None,
        depends_on: list[int] | None = None,
        is_optional: bool = False,
    ) -> int:
        step = ToolPlanStep(
            tool_name=tool_name,
            reason=reason,
            arguments=arguments or {},
            condition=condition,
            depends_on=depends_on or [],
            is_optional=is_optional,
        )
        self.steps.append(step)
        return len(self.steps) - 1

    def get_required_steps(self) -> list[ToolPlanStep]:
        return [step for step in self.steps if not step.is_optional]

    def get_optional_steps(self) -> list[ToolPlanStep]:
        return [step for step in self.steps if step.is_optional]