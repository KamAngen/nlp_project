from __future__ import annotations

import json
from typing import Any

from legal_agent.evaluation.metrics import citation_hit_rate, exact_match_score, token_f1
from legal_agent.models.qwen_local import LocalQwenChatModel


class HeuristicJudge:
    def judge(self, answer: str, expected_answer: str, references: list[str]) -> dict[str, Any]:
        exact_match = exact_match_score(answer, expected_answer)
        answer_f1 = token_f1(answer, expected_answer)
        citation_score = citation_hit_rate(answer, references)
        is_correct = exact_match > 0 or answer_f1 >= 0.55 or (
            answer_f1 >= 0.35 and citation_score is not None and citation_score > 0
        )
        return {
            "is_correct": bool(is_correct),
            "exact_match": exact_match,
            "answer_f1": answer_f1,
            "citation_score": citation_score,
        }


class LocalQwenJudge:
    def __init__(self, model: LocalQwenChatModel) -> None:
        self.model = model

    def judge(self, answer: str, expected_answer: str, references: list[str]) -> dict[str, Any]:
        prompt = {
            "role": "user",
            "content": (
                "请作为法律答案评审，判断模型答案是否与参考答案在核心法律结论上一致。"
                "请只输出 JSON：{\"is_correct\": true/false, \"reason\": \"...\"}。\n\n"
                f"参考答案：{expected_answer}\n\n模型答案：{answer}\n\n参考法条：{references}"
            ),
        }
        output = self.model.generate(
            [{"role": "system", "content": "你是严格的 JSON 评审器。"}, prompt],
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            top_k=20,
            presence_penalty=1.0,
            enable_thinking=False,
        )
        text = (output.content or output.raw_text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"is_correct": False, "reason": "judge_parse_failed"}
