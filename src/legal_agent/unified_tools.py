from __future__ import annotations

import json
from typing import Any

from legal_agent.agent.tools import ToolRegistry, ToolSpec
from legal_agent.study_tools import StudyToolExecutor


class UnifiedToolRegistry(ToolRegistry):
    def __init__(
        self,
        retriever,
        *,
        study_tool_executor: StudyToolExecutor,
        user_id: str,
        session_id: str,
        scripted_answers: dict[str, str] | None = None,
        interactive: bool = False,
        ask_user_handler=None,
    ) -> None:
        super().__init__(
            retriever,
            scripted_answers=scripted_answers,
            interactive=interactive,
            ask_user_handler=ask_user_handler,
        )
        self.study_tool_executor = study_tool_executor
        self.user_id = user_id
        self.session_id = session_id
        self._legacy_tools = set(self.specs)
        for spec in study_tool_executor.specs.values():
            self.specs[spec.name] = ToolSpec(
                name=spec.name,
                description=spec.description,
                parameters=dict(spec.input_schema),
                return_format=self._render_return_format(spec.output_schema),
            )

    def bind_session(self, *, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id

    def execute(self, tool_name: str, tool_args: dict[str, Any] | None) -> dict[str, Any]:
        if tool_name in self._legacy_tools:
            return super().execute(tool_name, tool_args)
        return self.study_tool_executor.execute(
            tool_name,
            tool_args or {},
            user_id=self.user_id,
            session_id=self.session_id,
        )

    @staticmethod
    def _render_return_format(output_schema: dict[str, str]) -> str:
        if not output_schema:
            return "JSON 对象"
        ordered = {key: output_schema[key] for key in sorted(output_schema)}
        return f"JSON 对象，字段：{json.dumps(ordered, ensure_ascii=False)}"
