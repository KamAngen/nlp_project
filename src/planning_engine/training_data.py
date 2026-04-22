from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from planning_engine.schema import ActionPlan, ToolPlanStep


@dataclass(slots=True)
class PlannerTrainingExample:
    query: str
    context_summary: str
    intent: str
    target_tools: list[str]
    plan_payload: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_planner_training_example(query: str, context_summary: str, plan: ActionPlan) -> PlannerTrainingExample:
    return PlannerTrainingExample(
        query=query,
        context_summary=context_summary,
        intent=plan.intent,
        target_tools=[step.tool_name for step in plan.steps],
        plan_payload=plan.to_dict(),
        notes=list(plan.notes),
    )


def build_planner_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        plan_payload = dict(row.get("plan") or {})
        plan = ActionPlan(
            intent=str(plan_payload.get("intent") or "legal_qa"),
            objective=str(plan_payload.get("objective") or ""),
            steps=[],
            response_style=str(plan_payload.get("response_style") or "structured"),
            notes=[str(item) for item in plan_payload.get("notes", [])],
            metadata=dict(plan_payload.get("metadata") or {}),
        )
        for step in plan_payload.get("steps", []):
            plan.steps.append(
                ToolPlanStep(
                    tool_name=str(step.get("tool_name") or ""),
                    reason=str(step.get("reason") or ""),
                    arguments=dict(step.get("arguments") or {}),
                )
            )
        example = build_planner_training_example(
            query=str(row.get("query") or ""),
            context_summary=str(row.get("context_summary") or ""),
            plan=plan,
        )
        examples.append(example.to_dict())
    return examples