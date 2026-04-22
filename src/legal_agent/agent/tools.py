from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass
import json
import re
from typing import Any, Callable

from legal_agent.data.admin_divisions import compare_region_relation
from legal_agent.rag.retriever import HybridLegalRetriever
from legal_agent.utils.text import truncate_text


ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}

LOCATION_CLARIFICATION_SUFFIX = "如果知道更具体位置，也可以直接补充到街道。"


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    return_format: str


def _resolved_query_location_payload(query_context: Any) -> dict[str, Any] | None:
    resolution = getattr(query_context, "location_resolution", None)
    if resolution is None:
        return None
    return {
        "province": resolution.province_name,
        "city": resolution.city_name,
        "county": resolution.county_name,
        "town": resolution.town_name,
        "village": resolution.village_name,
        "detail": resolution.detail,
        "full_location": resolution.full_location,
        "matched_level": resolution.explicit_level,
        "resolved_region_path": list(resolution.path_names),
    }


def _clarification_example_regions(labels: list[str]) -> str:
    if not labels:
        return ""
    preview = "、".join(labels[:3])
    return f"例如 {preview}。"


def build_location_clarification(query_context: Any, hits: list[Any]) -> dict[str, Any]:
    local_hits = [hit for hit in hits if hit.jurisdiction_type == "local" and hit.region_path_names]
    if not local_hits:
        return {
            "needs_location_clarification": False,
            "location_clarification_question": None,
        }

    if not getattr(query_context, "has_explicit_region", False):
        distinct_regions = {tuple(hit.region_path_names) for hit in local_hits}
        if len(distinct_regions) >= 2 and (
            getattr(query_context, "likely_local_question", False) or len(local_hits) >= 2
        ):
            labels = [" > ".join(hit.region_path_names) for hit in local_hits[:3]]
            return {
                "needs_location_clarification": True,
                "location_clarification_question": (
                    "该问题可能受地方性法规影响。请补充你所在的省、市、区县；"
                    f"{LOCATION_CLARIFICATION_SUFFIX} {_clarification_example_regions(labels)}"
                ).strip(),
            }
        return {
            "needs_location_clarification": False,
            "location_clarification_question": None,
        }

    explicit_level = getattr(query_context, "explicit_region_level", None)
    if explicit_level not in {"province", "prefecture"}:
        return {
            "needs_location_clarification": False,
            "location_clarification_question": None,
        }

    descendant_hits = []
    for hit in local_hits:
        relation = compare_region_relation(getattr(query_context, "regions", []), hit.region_path_codes)
        if relation == "descendant":
            descendant_hits.append(hit)
    if not descendant_hits:
        return {
            "needs_location_clarification": False,
            "location_clarification_question": None,
        }

    resolution = getattr(query_context, "location_resolution", None)
    anchor = None
    if resolution is not None:
        anchor = resolution.full_location or resolution.province_name or resolution.city_name
    if not anchor:
        anchor = "你目前提供的地点"

    if explicit_level == "province":
        refinement = "所在市、区县"
        descendant_labels = list(dict.fromkeys(
            hit.region_path_names[1] if len(hit.region_path_names) >= 2 else (hit.region_name or "")
            for hit in descendant_hits
            if hit.region_path_names or hit.region_name
        ))
    else:
        refinement = "所在区县"
        descendant_labels = list(dict.fromkeys(
            hit.region_path_names[-1] if hit.region_path_names else (hit.region_name or "")
            for hit in descendant_hits
            if hit.region_path_names or hit.region_name
        ))

    question = (
        f"你目前提供的地点是 {anchor}。该范围内可能存在更细粒度的地方性规定，"
        f"请补充 {refinement}；{LOCATION_CLARIFICATION_SUFFIX} {_clarification_example_regions(descendant_labels)}"
    ).strip()
    return {
        "needs_location_clarification": True,
        "location_clarification_question": question,
    }


class SafeCalculator:
    _percent_re = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*[%％]")
    _unit_re = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[亿万千百])")
    _currency_re = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(元|块钱|块)")

    def _normalize_expression(self, expression: str) -> str:
        normalized = str(expression).strip()
        normalized = normalized.replace("（", "(").replace("）", ")")
        normalized = normalized.replace("，", ",").replace("。", ".")
        normalized = normalized.replace("×", "*").replace("x", "*").replace("X", "*")
        normalized = normalized.replace("÷", "/")
        normalized = normalized.replace("＋", "+").replace("－", "-")
        normalized = normalized.replace("^", "**")
        normalized = normalized.replace("倍", "")
        normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalized)
        normalized = self._percent_re.sub(lambda match: f"({match.group('number')}/100)", normalized)
        normalized = self._currency_re.sub(lambda match: match.group("number"), normalized)
        unit_multiplier = {"亿": 100000000, "万": 10000, "千": 1000, "百": 100}
        normalized = self._unit_re.sub(
            lambda match: f"({match.group('number')}*{unit_multiplier[match.group('unit')]})",
            normalized,
        )
        return normalized

    def evaluate(self, expression: str) -> float:
        node = ast.parse(self._normalize_expression(expression), mode="eval")
        return float(self._eval_node(node.body))

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            args = [self._eval_node(arg) for arg in node.args]
            if node.func.id == "round" and len(args) == 2:
                return float(round(args[0], int(args[1])))
            return float(ALLOWED_FUNCTIONS[node.func.id](*args))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


class ToolRegistry:
    def __init__(
        self,
        retriever: HybridLegalRetriever,
        *,
        scripted_answers: dict[str, str] | None = None,
        interactive: bool = False,
        ask_user_handler: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.retriever = retriever
        self.calculator = SafeCalculator()
        self.scripted_answers = scripted_answers or {}
        self.interactive = interactive
        self.ask_user_handler = ask_user_handler
        self.specs: dict[str, ToolSpec] = {
            "retrieve_from_kb": ToolSpec(
                name="retrieve_from_kb",
                description="从本地法规知识库中检索最相关的法条片段，并返回带标题、条款和文本的结果。",
                parameters={"query": "str", "top_k": "int", "effect_level": "str | null"},
                return_format="JSON 列表，每项含标题、条款、效力层级、正文",
            ),
            "lookup_statute": ToolSpec(
                name="lookup_statute",
                description="按法规标题进行精确或近似查询，返回法规元数据和条文预览。",
                parameters={"title": "str"},
                return_format="JSON 对象，含标题、效力层级、日期和预览条文",
            ),
            "resolve_hierarchy": ToolSpec(
                name="resolve_hierarchy",
                description="说明法规的效力层级，并返回层级顺序。",
                parameters={"title_or_category": "str"},
                return_format="JSON 对象，含效力层级与排序",
            ),
            "calculator": ToolSpec(
                name="calculator",
                description="执行本地安全算术表达式计算。",
                parameters={"expression": "str"},
                return_format="JSON 对象，含 result 数值",
            ),
            "ask_user": ToolSpec(
                name="ask_user",
                description="在事实不足时向用户追问缺失信息。批处理模式下将读取脚本化答案。",
                parameters={"question": "str", "field_name": "str"},
                return_format="JSON 对象，含 answer 文本",
            ),
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "return_format": spec.return_format,
            }
            for spec in self.specs.values()
        ]

    def execute(self, tool_name: str, tool_args: dict[str, Any] | None) -> dict[str, Any]:
        tool_args = tool_args or {}
        if tool_name == "retrieve_from_kb":
            return self._retrieve_from_kb(**tool_args)
        if tool_name == "lookup_statute":
            return self._lookup_statute(**tool_args)
        if tool_name == "resolve_hierarchy":
            return self._resolve_hierarchy(**tool_args)
        if tool_name == "calculator":
            return self._calculator(**tool_args)
        if tool_name == "ask_user":
            return self._ask_user(**tool_args)
        raise KeyError(f"Unsupported tool: {tool_name}")

    def _retrieve_from_kb(self, query: str, top_k: int = 5, effect_level: str | None = None) -> dict[str, Any]:
        query_context = self.retriever.inspect_query(query)
        hits = self.retriever.search(query, top_k=top_k, effect_level=effect_level, query_context=query_context)
        clarification = build_location_clarification(query_context, hits)
        return {
            "query": query,
            "explicit_query_regions": [region.name for region in query_context.regions],
            "explicit_region_level": query_context.explicit_region_level,
            "resolved_query_location": _resolved_query_location_payload(query_context),
            "needs_location_clarification": clarification["needs_location_clarification"],
            "location_clarification_question": clarification["location_clarification_question"],
            "results": [
                {
                    "chunk_id": hit.chunk_id,
                    "document_title": hit.document_title,
                    "article_heading": hit.article_heading,
                    "effect_level": hit.effect_level,
                    "jurisdiction_type": hit.jurisdiction_type,
                    "jurisdiction_scope": hit.jurisdiction_scope,
                    "region_name": hit.region_name,
                    "region_path_names": hit.region_path_names,
                    "applies_to": " > ".join(hit.region_path_names) if hit.region_path_names else "全国",
                    "text": truncate_text(hit.text, self.retriever.config.retrieval.observation_max_chars),
                    "source_path": hit.source_path,
                    "score": round(hit.score, 4),
                }
                for hit in hits
            ],
        }

    def _lookup_statute(self, title: str) -> dict[str, Any]:
        result = self.retriever.lookup_statute(title)
        if result is None:
            raise ValueError(f"未找到法规标题: {title}")
        return result

    def _resolve_hierarchy(self, title_or_category: str) -> dict[str, Any]:
        return self.retriever.resolve_hierarchy(title_or_category)

    def _calculator(self, expression: str) -> dict[str, Any]:
        return {"expression": expression, "result": self.calculator.evaluate(expression)}

    def _ask_user(self, question: str, field_name: str = "") -> dict[str, Any]:
        if field_name and field_name in self.scripted_answers:
            return {"question": question, "answer": self.scripted_answers[field_name], "source": "scripted"}
        if question in self.scripted_answers:
            return {"question": question, "answer": self.scripted_answers[question], "source": "scripted"}
        if self.ask_user_handler is not None:
            answer = self.ask_user_handler(question, field_name)
            if answer is None or not str(answer).strip():
                return {
                    "question": question,
                    "field_name": field_name,
                    "status": "pending_user_input",
                }
            return {"question": question, "answer": str(answer).strip(), "source": "callback"}
        if not self.interactive:
            raise ValueError(f"缺少脚本化答案，且当前不是交互模式: {field_name or question}")
        answer = input(f"[ASK_USER] {question}\n> ").strip()
        return {"question": question, "answer": answer, "source": "interactive"}


def observation_to_text(payload: dict[str, Any]) -> str:
    compact = dict(payload)
    if isinstance(compact.get("results"), list):
        compact_results = []
        for item in compact["results"][:4]:
            compact_item = dict(item)
            if "text" in compact_item:
                compact_item["text"] = truncate_text(str(compact_item.get("text") or ""), 320)
            compact_results.append(compact_item)
        compact["results"] = compact_results
    if isinstance(compact.get("questions"), list):
        compact_questions = []
        for item in compact["questions"][:3]:
            compact_item = dict(item)
            if "question" in compact_item:
                compact_item["question"] = truncate_text(str(compact_item.get("question") or ""), 180)
            if isinstance(compact_item.get("analysis"), str):
                compact_item["analysis"] = truncate_text(str(compact_item.get("analysis") or ""), 180)
            compact_questions.append(compact_item)
        compact["questions"] = compact_questions
    if isinstance(compact.get("details"), list):
        compact_details = []
        for item in compact["details"][:3]:
            compact_item = dict(item)
            if isinstance(compact_item.get("analysis"), str):
                compact_item["analysis"] = truncate_text(str(compact_item.get("analysis") or ""), 180)
            if isinstance(compact_item.get("question"), str):
                compact_item["question"] = truncate_text(str(compact_item.get("question") or ""), 180)
            compact_details.append(compact_item)
        compact["details"] = compact_details
    if isinstance(compact.get("preview_texts"), list):
        compact["preview_texts"] = [truncate_text(str(text or ""), 240) for text in compact["preview_texts"][:3]]
    if isinstance(compact.get("report_markdown"), str):
        compact["report_markdown"] = truncate_text(str(compact.get("report_markdown") or ""), 480)
    return json.dumps(compact, ensure_ascii=False, indent=2)
