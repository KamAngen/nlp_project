from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from heapq import heappush, heapreplace
import json
from pathlib import Path
import random
import re
from typing import Any

from legal_agent.config import AppConfig
from legal_agent.data.disc_law import download_disc_law_dataset, normalize_disc_law_dataset
from legal_agent.study_config import StudyAgentConfig
from legal_agent.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from legal_agent.utils.text import clean_text, simple_tokenize, truncate_text


REFERENCE_TITLE_RE = re.compile(r"《([^》\n]{2,120})》")
OPTION_LABEL_RE = re.compile(r"(?<![A-Za-z])[A-D][.．、]\s*")
CHOICE_MARKER_RE = re.compile(r"(?<![A-Za-z])([A-D])[.．、]\s*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,：:])\s*")
NON_WORD_RE = re.compile(r"[\s，,。；;：:！!？?（）()《》“”\"'、]+")
TOPIC_ORDER = ("民法", "刑法", "行政法", "民诉", "刑诉", "商经", "理论法")
CHOICE_STEM_HINTS = (
    "下列",
    "以下",
    "哪一",
    "何种",
    "正确",
    "错误",
    "是否",
    "请问",
    "可以",
    "应当",
    "如何",
    "处理",
    "属于",
    "说法",
    "判断",
    "结果",
    "选项",
    "?",
    "？",
)
QUESTION_PREFIXES = (
    "请给出以下单选问题的答案和推理过程：",
    "请给出这道单选题的答案和推理过程。",
    "请给出这道单选题的答案和推理过程：",
    "请给出单选题的答案和解析。",
    "请给出单选题的答案和解析：",
    "请给出单选题的答案。",
    "请给出答案和解析。",
    "请给出答案和解析：",
    "以下是一道单项选择题：",
    "以下是一道单选题：",
    "这是一道单选题。",
    "这是一道单选题：",
    "单选：",
    "单项选择题：",
    "问题：",
    "问题:",
)
QUESTION_SUFFIX_MARKERS = (
    "请给出答案和解析",
    "请给出答案",
    "请给出详细的推理过程之后再给出答案",
    "请给出推理过程",
    "给出详细的推理过程之后再给出答案",
    "请一步步思考，选择合适的选项并给出理由",
    "请一步步思考并给出理由",
    "请一步步思考",
)
QUESTION_REJECT_HINTS = ("多选", "多项", "下列哪些", "哪些说法", "哪些情形", "哪几项", "哪几种")
ANSWER_PREFIXES = ("正确答案：", "正确答案:", "参考答案：", "参考答案:", "答案：", "答案:")
ANSWER_SPLIT_MARKERS = ("解析：", "解析:", "参考解析：", "参考解析:", "理由：", "理由:")
ANSWER_LABEL_PATTERNS = (
    re.compile(r"^\s*(?:答案(?:为)?|应选|选择|选|正确的是|正确选项是)?[:：]?\s*([A-D])(?:[.．、\s]|$)"),
    re.compile(r"选项\s*([A-D])"),
    re.compile(r"答案(?:为)?\s*([A-D])"),
)
QUESTION_TASK_FAMILIES = {
    "legal_question_answering",
    "exam",
    "jud_read_compre",
    "judgement_predit",
    "sent_pred",
}
CASE_TASK_FAMILIES = {
    "legal_question_answering",
    "jud_read_compre",
    "leg_case_cls",
    "sim_case_match",
    "jud_doc_sum",
}
TOPIC_KEYWORDS = {
    "民法": ("民法", "民法典", "合同", "租赁", "物权", "侵权", "婚姻", "继承", "担保", "押金"),
    "刑法": ("刑法", "诈骗", "盗窃", "抢劫", "故意", "犯罪", "刑罚", "交通肇事", "受贿", "量刑"),
    "行政法": ("行政法", "行政处罚", "行政许可", "行政复议", "行政机关", "听证", "行政赔偿", "强制", "政府"),
    "民诉": ("民事诉讼", "民诉", "管辖", "起诉", "保全", "执行", "证据", "再审", "调解"),
    "刑诉": ("刑事诉讼", "刑诉", "侦查", "起诉", "审判", "辩护", "取保候审", "证据排除", "羁押"),
    "商经": ("公司法", "证券法", "破产法", "保险法", "票据", "税法", "合伙", "商经", "反垄断", "消费者"),
    "理论法": ("宪法", "法理", "立法法", "法律效力", "法律原则", "法治", "法制史", "司法制度", "理论法"),
}
GENERIC_DISTRACTORS = {
    "民法": [
        "通常只能请求行政机关先作出处罚，不能直接主张民事救济。",
        "只要存在口头约定，就可以排除书面合同和法定规则的适用。",
        "原则上当然无效，不需要再结合合同约定和损失情况判断。",
        "争议一律转化为刑事责任评价，不再审查民事请求基础。",
    ],
    "刑法": [
        "只要造成损失，就一律按民事侵权处理，不再进入刑法评价。",
        "只需看行为后果，不需要审查主观故意和构成要件。",
        "行为一经发生就当然按最重罪名评价，不作区分。",
        "只涉及行政处罚，不影响刑事责任认定。",
    ],
    "行政法": [
        "行政机关作出处分前无需再审查程序性保障是否到位。",
        "只要作出行政决定，当事人原则上不能再主张任何程序权利。",
        "程序瑕疵不影响行政行为评价，因此无需讨论听证或告知。",
        "只要行政机关认为必要，就可以排除法定程序限制。",
    ],
    "民诉": [
        "民事诉讼程序中原则上不需要审查管辖、证据或保全问题。",
        "当事人起诉后，法院当然应直接进入实体裁判。",
        "程序违法不会影响判决结果，因此无需复查程序节点。",
        "一旦立案，当事人就不能再主张任何程序保障。",
    ],
    "刑诉": [
        "刑事程序中侦查机关可以当然排除辩护和证据规则的约束。",
        "只要涉嫌犯罪，程序阶段保障就不再重要。",
        "程序违法通常不影响案件处理，因此无需继续审查。",
        "辩护权和取证规则只在判决后才发生作用。",
    ],
    "商经": [
        "商经法问题原则上都按刑事责任路径处理即可。",
        "只要是市场交易，就不需要区分公司、证券、破产等制度。",
        "企业内部决议当然优先于法律规范适用。",
        "只看商业惯例即可，不需要回到具体法条结构。",
    ],
    "理论法": [
        "法律冲突时无需审查效力层级，任选其一即可。",
        "规范适用只看发布时间，不必考虑上位法和特别法。",
        "法理类题目通常不涉及概念区分和适用顺序。",
        "只要存在司法解释，就当然高于法律。",
    ],
}
STATIC_COMMON_KNOWLEDGE = [
    {
        "entry_id": "study-method-001",
        "title": "法考作答结构",
        "content": "法考学习场景下，优先写结论，再写法律依据，最后补充展开分析和例外情形。",
        "tags": ["答题模板", "学习方法", "主观题"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-002",
        "title": "错题复盘四问",
        "content": "复盘错题时，至少记录法条是否熟、概念是否混淆、程序阶段是否记错、审题是否跑偏四个维度。",
        "tags": ["错题", "复盘", "学习方法"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-003",
        "title": "法规检索关键词",
        "content": "检索法规时，先用主体、行为、后果和程序节点四类关键词缩小范围，再核对法条标题与效力层级。",
        "tags": ["法规检索", "学习方法", "关键词"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-004",
        "title": "程序法刷题顺序",
        "content": "程序法题先看程序阶段，再看主体权限、权利保障和救济路径，最后再判断结论。",
        "tags": ["程序法", "学习方法", "刷题"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-005",
        "title": "法条记忆优先级",
        "content": "背法条时优先记适用条件、关键词、主体范围和例外，再记数字、期限和程序细节。",
        "tags": ["法条背诵", "记忆策略", "学习方法"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-006",
        "title": "案例分析拆题法",
        "content": "分析案例题时，先拆事实，再找争点，随后分别写依据和结论，不要把多个争点混在一段里。",
        "tags": ["案例分析", "学习方法", "答题模板"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-007",
        "title": "单选题排除法",
        "content": "做单选题时先排除绝对化、跳步推理和明显换法域的选项，再在剩余选项里比较法条依据和程序顺序。",
        "tags": ["单选题", "排除法", "学习方法"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-008",
        "title": "程序题审题顺序",
        "content": "程序题先定位阶段，再看主体权限、告知听证、救济路径和期限，最后判断结论是否成立。",
        "tags": ["程序法", "审题", "学习方法"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-009",
        "title": "法条依据书写法",
        "content": "写依据时优先给法条方向和判断要件，不必机械堆砌全文；重点写主体、条件、后果和例外。",
        "tags": ["法条依据", "答题模板", "学习方法"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-method-010",
        "title": "错因分类法",
        "content": "错题至少分成法条不熟、概念混淆、程序阶段记错、选项比较失误四类，复盘时按类别集中处理。",
        "tags": ["错题", "复盘", "学习方法"],
        "source": "auto_builder",
    },
]
TOPIC_COMMON_GUIDES = {
    "民法": {
        "keywords": ["合同", "物权", "侵权"],
        "pitfall": "容易把民事责任、行政责任和刑事责任混在一起，审题时先分清请求权基础。",
    },
    "刑法": {
        "keywords": ["构成要件", "故意过失", "罪数"],
        "pitfall": "容易只看结果不看主观要素和行为阶段，判断时先看构成要件再看量刑情节。",
    },
    "行政法": {
        "keywords": ["行政处罚", "行政许可", "程序保障"],
        "pitfall": "容易忽略告知、听证、复议诉讼等程序节点，做题时先看程序再看实体权限。",
    },
    "民诉": {
        "keywords": ["管辖", "证据", "执行"],
        "pitfall": "容易把立案、审理、执行和再审阶段混用，先定程序阶段再判断权利义务。",
    },
    "刑诉": {
        "keywords": ["侦查", "辩护", "证据规则"],
        "pitfall": "容易忽略辩护权、强制措施和非法证据排除，程序保障通常是高频失分点。",
    },
    "商经": {
        "keywords": ["公司治理", "证券规则", "票据责任"],
        "pitfall": "容易把公司内部关系和对外责任混在一起，先分主体身份再看责任承担路径。",
    },
    "理论法": {
        "keywords": ["效力层级", "法理原则", "立法权限"],
        "pitfall": "容易把宪法、立法法和法理概念混写，答题时先定层级关系再写适用规则。",
    },
}
GLOBAL_COMMON_GUIDES = [
    {
        "entry_id": "study-global-001",
        "title": "综合练习使用方式",
        "content": "综合练习更适合查弱项分布，不要只看总分；应同时记录错题主题、程序节点和常错选项类型。",
        "tags": ["综合练习", "复盘", "学习方法"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-global-002",
        "title": "法条检索优先级",
        "content": "检索依据时优先找上位法和主干法条，再补司法解释、程序规则和地方规范，避免一开始陷入细枝末节。",
        "tags": ["法规检索", "学习方法", "法条体系"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-global-003",
        "title": "模拟测试复盘模板",
        "content": "每次模拟测试后至少输出三项：错题主题、错误原因、下一轮复习动作，这样画像和报告才有持续价值。",
        "tags": ["模拟测试", "复盘", "学习报告"],
        "source": "auto_builder",
    },
    {
        "entry_id": "study-global-004",
        "title": "案例检索使用提示",
        "content": "案例库更适合找争点拆解和说理模板，不适合直接替代法条；用案例时要同时回到对应制度和程序节点。",
        "tags": ["案例库", "检索", "学习方法"],
        "source": "auto_builder",
    },
]
STATIC_SYSTEM_MEMORIES = [
    {
        "id": "system-style-01",
        "category": "response_policy",
        "text": "法考学习场景下，回答应先给简短结论，再给知识依据，再给学习建议，避免直接堆砌长法条。",
        "importance": 0.96,
        "tags": ["reply_style", "study_agent"],
    },
    {
        "id": "system-memory-01",
        "category": "memory_policy",
        "text": "用户画像中的备考科目、薄弱点、学习节奏和目标分数属于高重要长期记忆，不应被普通对话轻易覆盖。",
        "importance": 0.94,
        "tags": ["memory", "profile"],
    },
    {
        "id": "system-exam-01",
        "category": "exam_policy",
        "text": "模拟测试默认按百分制展示成绩，并在评分后提炼薄弱知识点，写回用户画像与学习报告。",
        "importance": 0.93,
        "tags": ["exam", "report"],
    },
    {
        "id": "system-exam-02",
        "category": "wrong_question_policy",
        "text": "错题库中的题目应在后续练习中被随机回放，答对后再从错题库移除。",
        "importance": 0.92,
        "tags": ["exam", "wrong_question_bank"],
    },
    {
        "id": "system-retrieval-01",
        "category": "retrieval_policy",
        "text": "学习问答应优先综合题库解析、案例要点和常识知识；需要法规依据时，再补充法条检索结果。",
        "importance": 0.91,
        "tags": ["retrieval", "study_agent"],
    },
    {
        "id": "system-report-01",
        "category": "report_policy",
        "text": "学习报告至少应概括当前画像、最近一次测试、错题库状态和下一步复习建议。",
        "importance": 0.9,
        "tags": ["report", "study_agent"],
    },
]


@dataclass(slots=True)
class StudyCandidate:
    sample_id: str
    topic: str
    question: str
    answer: str
    answer_summary: str
    analysis: str
    references: list[str]
    tags: list[str]
    task_family: str
    difficulty: str
    source_options: dict[str, str] | None = None
    source_answer_label: str | None = None


@dataclass(slots=True)
class ParsedChoiceQuestion:
    stem: str
    options: dict[str, str]
    answer_label: str


def prepare_study_knowledge_assets(
    app_config: AppConfig,
    study_config: StudyAgentConfig,
    *,
    question_count: int = 180,
    case_count: int = 96,
    common_count: int = 24,
    force_rebuild: bool = False,
    auto_download_disc_law: bool = False,
) -> dict[str, Any]:
    existing = _study_asset_status(study_config)
    required_system_count = max(len(STATIC_SYSTEM_MEMORIES), 6)
    if (
        not force_rebuild
        and existing["question_count"] >= question_count
        and existing["case_count"] >= case_count
        and existing["common_count"] >= common_count
        and existing["system_count"] >= required_system_count
    ):
        return {"generated": False, **existing}

    disc_path = _ensure_disc_law_normalized(app_config, auto_download_disc_law=auto_download_disc_law)
    catalog_rows = _read_law_catalog_rows(app_config)

    question_pools, case_pools = _collect_candidate_pools(
        disc_path,
        question_capacity_per_topic=max(question_count, 72),
        case_capacity_per_topic=max(case_count, 48),
    )
    question_candidates = _select_candidates(question_pools, question_count + max(24, question_count // 4))
    case_candidates = _select_candidates(case_pools, case_count + max(12, case_count // 4))

    preserved_questions = _load_preserved_jsonl_rows(study_config.question_bank_path, "question_id", "auto-q-")
    preserved_cases = _load_preserved_jsonl_rows(study_config.case_bank_path, "case_id", "auto-case-")
    preserved_common = _load_preserved_jsonl_rows(study_config.common_knowledge_path, "entry_id", "auto-common-")
    preserved_system = _load_preserved_system_rows(study_config.system_memory_path, "auto-system-")

    question_rows = _build_question_bank_rows(question_candidates, preserved_questions, question_target=question_count)
    case_rows = _build_case_bank_rows(case_candidates, preserved_cases, case_target=case_count)
    common_rows = _build_common_knowledge_rows(
        question_candidates,
        case_candidates,
        catalog_rows,
        preserved_common,
        common_target=common_count,
    )
    system_rows = _build_system_seed_rows(preserved_system)

    ensure_dir(study_config.question_bank_path.parent)
    write_jsonl(study_config.question_bank_path, question_rows)
    write_jsonl(study_config.case_bank_path, case_rows)
    write_jsonl(study_config.common_knowledge_path, common_rows)
    write_json(study_config.system_memory_path, system_rows)

    topic_counts = Counter(row.get("topic") or "综合" for row in question_rows)
    return {
        "generated": True,
        "question_count": len(question_rows),
        "case_count": len(case_rows),
        "common_count": len(common_rows),
        "system_count": len(system_rows),
        "question_bank_path": str(study_config.question_bank_path),
        "case_bank_path": str(study_config.case_bank_path),
        "common_knowledge_path": str(study_config.common_knowledge_path),
        "system_memory_path": str(study_config.system_memory_path),
        "topic_distribution": {topic: topic_counts[topic] for topic in TOPIC_ORDER if topic_counts[topic]},
        "catalog_row_count": len(catalog_rows),
        "disc_law_normalized_path": str(disc_path),
    }


def _study_asset_status(study_config: StudyAgentConfig) -> dict[str, int]:
    return {
        "question_count": _count_jsonl_rows(study_config.question_bank_path),
        "case_count": _count_jsonl_rows(study_config.case_bank_path),
        "common_count": _count_jsonl_rows(study_config.common_knowledge_path),
        "system_count": _count_json_rows(study_config.system_memory_path),
    }


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _count_json_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(read_json(path))
    except Exception:
        return 0


def _ensure_disc_law_normalized(app_config: AppConfig, *, auto_download_disc_law: bool) -> Path:
    ensure_dir(app_config.disc_law_dir)
    if app_config.disc_law_normalized_path.exists():
        return app_config.disc_law_normalized_path
    if app_config.disc_law_raw_dir.exists() and list(app_config.disc_law_raw_dir.glob("*.jsonl")):
        normalize_disc_law_dataset(app_config.disc_law_raw_dir, app_config.disc_law_normalized_path)
        return app_config.disc_law_normalized_path
    if auto_download_disc_law:
        download_disc_law_dataset(app_config.disc_law_raw_dir)
        normalize_disc_law_dataset(app_config.disc_law_raw_dir, app_config.disc_law_normalized_path)
        return app_config.disc_law_normalized_path
    raise FileNotFoundError(
        "未找到 DISC-Law 规范化数据。请先执行 bash scripts/download_disc_law.sh，或使用 --auto-download-disc-law。"
    )


def _read_law_catalog_rows(app_config: AppConfig) -> list[dict[str, str]]:
    catalog_path = app_config.law_dir / app_config.law_catalog_glob
    if not catalog_path.exists():
        return []
    with catalog_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): str(value or "").strip() for key, value in row.items()} for row in reader]


def _collect_candidate_pools(
    disc_path: Path,
    *,
    question_capacity_per_topic: int,
    case_capacity_per_topic: int,
) -> tuple[dict[str, list[StudyCandidate]], dict[str, list[StudyCandidate]]]:
    question_heaps: dict[str, list[tuple[int, int, StudyCandidate]]] = defaultdict(list)
    case_heaps: dict[str, list[tuple[int, int, StudyCandidate]]] = defaultdict(list)
    sequence = 0

    with disc_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            candidate = _candidate_from_disc_record(row)
            if candidate is None:
                continue
            score = _stable_score(candidate.sample_id)
            if candidate.task_family in QUESTION_TASK_FAMILIES:
                _push_candidate(question_heaps[candidate.topic], score, sequence, candidate, question_capacity_per_topic)
                sequence += 1
            if candidate.task_family in CASE_TASK_FAMILIES:
                _push_candidate(case_heaps[candidate.topic], score, sequence, candidate, case_capacity_per_topic)
                sequence += 1

    return _finalize_candidate_heaps(question_heaps), _finalize_candidate_heaps(case_heaps)


def _push_candidate(
    heap: list[tuple[int, int, StudyCandidate]],
    score: int,
    sequence: int,
    candidate: StudyCandidate,
    capacity: int,
) -> None:
    entry = (-score, sequence, candidate)
    if len(heap) < capacity:
        heappush(heap, entry)
        return
    if score < -heap[0][0]:
        heapreplace(heap, entry)


def _finalize_candidate_heaps(
    heaps: dict[str, list[tuple[int, int, StudyCandidate]]],
) -> dict[str, list[StudyCandidate]]:
    finalized: dict[str, list[StudyCandidate]] = {}
    for topic, heap in heaps.items():
        ordered = sorted(heap, key=lambda item: (-item[0], item[1]))
        finalized[topic] = [candidate for _, _, candidate in ordered]
    return finalized


def _candidate_from_disc_record(row: dict[str, Any]) -> StudyCandidate | None:
    task_family = str(row.get("task_family") or "").strip()
    question = _extract_disc_question(str(row.get("instruction") or ""), task_family)
    answer = clean_text(str(row.get("answer") or ""))
    if len(question) < 8 or len(answer) < 16:
        return None
    if _should_skip_question_candidate(question, answer):
        return None

    parsed_choice = _parse_choice_question(question, answer)
    source_options: dict[str, str] | None = None
    source_answer_label: str | None = None
    if parsed_choice is not None:
        question = _compact_choice_question(parsed_choice.stem)
        if not question:
            return None
        answer_summary = parsed_choice.options[parsed_choice.answer_label]
        analysis = _build_analysis_text(answer, correct_text=answer_summary)
        source_options = parsed_choice.options
        source_answer_label = parsed_choice.answer_label
    else:
        question = _compact_freeform_question(question)
        if not question:
            return None
        answer_summary = _answer_summary(answer)
        analysis = _build_analysis_text(answer, correct_text=answer_summary)
    if len(answer_summary) < 12:
        return None
    if len(analysis) < 12:
        analysis = answer_summary
    references = _extract_reference_titles(list(row.get("references") or []))
    references.extend(_extract_reference_titles([answer]))
    references = list(dict.fromkeys(references))[:4]
    topic = _infer_topic(question, answer, references)
    if topic == "综合":
        return None
    tags = _build_tags(topic, question, answer, references)
    difficulty = _infer_difficulty(task_family, question, answer)
    return StudyCandidate(
        sample_id=str(row.get("sample_id") or "unknown"),
        topic=topic,
        question=question,
        answer=answer,
        answer_summary=answer_summary,
        analysis=analysis,
        references=references,
        tags=tags,
        task_family=task_family,
        difficulty=difficulty,
        source_options=source_options,
        source_answer_label=source_answer_label,
    )


def _extract_disc_question(instruction: str, task_family: str) -> str:
    cleaned = clean_text(instruction)
    if task_family == "legal_question_answering":
        for marker in ("<问题>：", "<问题>:", "问题：", "问题:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker)[-1].strip()
                break
    return _sanitize_question_text(cleaned)


def _sanitize_question_text(text: str) -> str:
    cleaned = clean_text(text)
    changed = True
    while changed:
        changed = False
        for prefix in QUESTION_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                changed = True
    for marker in QUESTION_SUFFIX_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, maxsplit=1)[0].strip()
    return clean_text(cleaned).strip(" ：:；;。")


def _compact_freeform_question(text: str, *, max_chars: int = 220) -> str:
    cleaned = _sanitize_question_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    for marker in ("问题：", "问题:", "问：", "问:", "请问"):
        if marker in cleaned:
            candidate = clean_text(cleaned.split(marker, maxsplit=1)[-1])
            if candidate and len(candidate) <= max_chars:
                return candidate
    for segment in _split_sentences(cleaned):
        if len(segment) <= max_chars and any(hint in segment for hint in CHOICE_STEM_HINTS):
            return segment
    return ""


def _compact_choice_question(text: str, *, max_chars: int = 520) -> str:
    cleaned = _sanitize_question_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    for marker in ("问题：", "问题:", "请问", "下列", "以下", "关于"):
        index = cleaned.rfind(marker)
        if index >= 0:
            candidate = clean_text(cleaned[index:])
            if candidate and len(candidate) <= max_chars:
                return candidate
    return ""


def _parse_choice_question(question: str, answer: str) -> ParsedChoiceQuestion | None:
    extracted = _extract_choice_options(question)
    if extracted is None:
        return None
    stem, options = extracted
    if not _looks_like_choice_stem(stem):
        return None
    answer_label = _extract_answer_label(answer, options)
    if answer_label is None:
        return None
    return ParsedChoiceQuestion(stem=stem, options=options, answer_label=answer_label)


def _extract_choice_options(question: str) -> tuple[str, dict[str, str]] | None:
    matches = list(CHOICE_MARKER_RE.finditer(question))
    if len(matches) < 4:
        return None
    labels = [match.group(1) for match in matches[:4]]
    if labels != ["A", "B", "C", "D"]:
        return None
    stem = question[: matches[0].start()].strip()
    options: dict[str, str] = {}
    for index, match in enumerate(matches[:4]):
        start = match.end()
        end = matches[index + 1].start() if index < 3 else len(question)
        option_text = _sanitize_option_text(question[start:end])
        if not option_text or len(option_text) > 220:
            return None
        options[match.group(1)] = option_text
    if len(set(options.values())) < 4:
        return None
    return stem, options


def _sanitize_option_text(text: str) -> str:
    cleaned = clean_text(text)
    for prefix in ANSWER_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    cleaned = OPTION_LABEL_RE.sub("", cleaned, count=1)
    for marker in QUESTION_SUFFIX_MARKERS + ANSWER_SPLIT_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, maxsplit=1)[0].strip()
    return clean_text(cleaned).strip(" ：:；;，,。")


def _looks_like_choice_stem(stem: str) -> bool:
    cleaned = clean_text(stem)
    return any(hint in cleaned for hint in CHOICE_STEM_HINTS)


def _extract_answer_label(answer: str, options: dict[str, str]) -> str | None:
    normalized = clean_text(answer)
    for pattern in ANSWER_LABEL_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(1)

    answer_key = _comparison_key(_sanitize_answer_summary(answer))
    if not answer_key:
        return None
    for label, option_text in options.items():
        option_key = _comparison_key(option_text)
        if not option_key:
            continue
        if answer_key == option_key or answer_key.startswith(option_key) or option_key in answer_key:
            return label
    return None


def _should_skip_question_candidate(question: str, answer: str) -> bool:
    if any(hint in question for hint in QUESTION_REJECT_HINTS):
        return True
    answer_head = clean_text(answer)
    for marker in ANSWER_SPLIT_MARKERS:
        if marker in answer_head:
            answer_head = answer_head.split(marker, maxsplit=1)[0].strip()
            break
    return len(OPTION_LABEL_RE.findall(answer_head)) > 1


def _comparison_key(text: str) -> str:
    return NON_WORD_RE.sub("", clean_text(text)).lower()


def _extract_reference_titles(references: list[str]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for ref in references:
        for match in REFERENCE_TITLE_RE.finditer(str(ref or "")):
            title = clean_text(match.group(1))
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
    return titles


def _infer_topic(question: str, answer: str, references: list[str]) -> str:
    corpus = " ".join([question, answer, *references])
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in corpus for keyword in keywords):
            return topic
    token_set = set(simple_tokenize(corpus))
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in token_set for keyword in keywords):
            return topic
    return "综合"


def _build_tags(topic: str, question: str, answer: str, references: list[str]) -> list[str]:
    tags = [topic]
    tags.extend(references[:2])
    corpus = question + " " + answer
    for keyword in TOPIC_KEYWORDS.get(topic, ()):
        if keyword in corpus and keyword not in tags:
            tags.append(keyword)
        if len(tags) >= 5:
            break
    return tags[:5]


def _infer_difficulty(task_family: str, question: str, answer: str) -> str:
    if task_family in {"exam", "jud_read_compre", "judgement_predit"} or len(question) > 70 or len(answer) > 180:
        return "hard"
    if task_family in {"legal_question_answering", "sent_pred", "sim_case_match"}:
        return "medium"
    return "easy"


def _answer_summary(answer: str, *, max_chars: int = 90) -> str:
    normalized = _sanitize_answer_summary(answer)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    summarized = _summarize_complete_text(normalized, max_chars=max_chars, max_sentences=1)
    if summarized:
        return summarized
    for marker in ("，", ",", "：", ":"):
        if marker in normalized:
            candidate = normalized.split(marker, maxsplit=1)[0].strip()
            if 12 <= len(candidate) <= max_chars:
                return candidate
    return ""


def _sanitize_answer_summary(answer: str) -> str:
    normalized = clean_text(answer)
    for prefix in ANSWER_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    for marker in ANSWER_SPLIT_MARKERS:
        if marker in normalized:
            normalized = normalized.split(marker, maxsplit=1)[0].strip()
            break
    normalized = OPTION_LABEL_RE.sub("", normalized, count=1)
    for marker in QUESTION_SUFFIX_MARKERS:
        if marker in normalized:
            normalized = normalized.split(marker, maxsplit=1)[0].strip()
    return clean_text(normalized).strip(" ：:；;。")


def _build_analysis_text(answer: str, *, correct_text: str | None = None, max_chars: int = 480) -> str:
    normalized = clean_text(answer)
    if not normalized:
        return clean_text(correct_text or "")

    for prefix in ANSWER_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    normalized = OPTION_LABEL_RE.sub("", normalized, count=1)

    explanation = normalized
    for marker in ANSWER_SPLIT_MARKERS:
        if marker in explanation:
            head, tail = explanation.split(marker, maxsplit=1)
            explanation = tail.strip() or head.strip()
            break

    if correct_text:
        explanation = _strip_answer_echo(explanation, correct_text)
    explanation = clean_text(explanation)
    if not explanation:
        explanation = clean_text(correct_text or "")

    summarized = _summarize_complete_text(
        explanation,
        max_chars=max_chars,
        max_sentences=3,
        allow_clause_split=False,
    )
    if summarized:
        return summarized

    if len(explanation) <= 1200:
        return explanation

    fallback = clean_text(correct_text or "")
    if fallback and len(fallback) <= max_chars:
        return fallback
    return ""


def _strip_answer_echo(text: str, correct_text: str) -> str:
    normalized = clean_text(text)
    expected = clean_text(correct_text)
    if expected and normalized.startswith(expected):
        stripped = normalized[len(expected) :].lstrip(" ，,；;：:。")
        if stripped:
            return stripped
    return normalized


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in SENTENCE_SPLIT_RE.split(clean_text(text)) if segment.strip()]


def _summarize_complete_text(
    text: str,
    *,
    max_chars: int,
    max_sentences: int,
    allow_clause_split: bool = True,
) -> str:
    normalized = clean_text(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized

    splitters = [SENTENCE_SPLIT_RE]
    if allow_clause_split:
        splitters.append(CLAUSE_SPLIT_RE)

    for splitter in splitters:
        parts = [part.strip() for part in splitter.split(normalized) if part.strip()]
        if not parts:
            continue
        selected: list[str] = []
        for part in parts:
            candidate = clean_text(" ".join([*selected, part]))
            if len(candidate) > max_chars:
                break
            selected.append(part)
            if len(selected) >= max_sentences:
                break
        if selected:
            return clean_text(" ".join(selected)).rstrip("，,；;：:")
    return ""


def _stable_score(value: str) -> int:
    digest = blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _select_candidates(pools: dict[str, list[StudyCandidate]], target_count: int) -> list[StudyCandidate]:
    if target_count <= 0:
        return []
    topics = [topic for topic in TOPIC_ORDER if pools.get(topic)]
    if not topics:
        return []
    targets: dict[str, int] = {topic: 0 for topic in topics}
    remaining = target_count
    while remaining > 0:
        progressed = False
        for topic in topics:
            if targets[topic] >= len(pools[topic]):
                continue
            targets[topic] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected: list[StudyCandidate] = []
    for topic in topics:
        selected.extend(pools[topic][: targets[topic]])
    if len(selected) >= target_count:
        return selected[:target_count]

    leftovers: list[StudyCandidate] = []
    for topic in topics:
        leftovers.extend(pools[topic][targets[topic] :])
    return (selected + leftovers)[:target_count]


def _load_preserved_jsonl_rows(path: Path, id_key: str, generated_prefix: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    preserved: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        identifier = str(row.get(id_key) or "")
        if identifier.startswith(generated_prefix):
            continue
        preserved.append(dict(row))
    return preserved


def _load_preserved_system_rows(path: Path, generated_prefix: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = read_json(path)
    except Exception:
        return []
    preserved: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row.get("id") or "")
        if identifier.startswith(generated_prefix):
            continue
        preserved.append(dict(row))
    return preserved


def _build_question_bank_rows(
    candidates: list[StudyCandidate],
    preserved_rows: list[dict[str, Any]],
    *,
    question_target: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in preserved_rows]
    seen_ids = {str(row.get("question_id") or "") for row in rows}
    seen_questions = {str(row.get("question") or "") for row in rows}

    for candidate in candidates:
        if len(rows) >= question_target:
            break
        question_id = _question_record_id(candidate)
        if question_id in seen_ids or candidate.question in seen_questions:
            continue
        if candidate.source_options and candidate.source_answer_label:
            options = dict(candidate.source_options)
            answer_label = candidate.source_answer_label
        else:
            options, answer_label = _build_question_options(candidate)
        row = {
            "question_id": question_id,
            "topic": candidate.topic,
            "difficulty": candidate.difficulty,
            "question": candidate.question,
            "options": options,
            "answer": answer_label,
            "analysis": candidate.analysis,
            "tags": candidate.tags,
            "score": 20,
        }
        rows.append(row)
        seen_ids.add(question_id)
        seen_questions.add(candidate.question)
    return rows


def _build_question_options(
    candidate: StudyCandidate,
) -> tuple[dict[str, str], str]:
    correct = candidate.answer_summary
    distractors: list[str] = []
    for text in GENERIC_DISTRACTORS.get(candidate.topic, []):
        if text != correct and text not in distractors:
            distractors.append(text)
        if len(distractors) >= 3:
            break
    while len(distractors) < 3:
        fallback = f"需要结合更具体案情继续判断，但这不是本题最优结论。{len(distractors) + 1}"
        if fallback not in distractors:
            distractors.append(fallback)

    rng = random.Random(_stable_score(candidate.sample_id))
    option_values = distractors[:3] + [correct]
    rng.shuffle(option_values)
    labels = ["A", "B", "C", "D"]
    options = {labels[index]: option_values[index] for index in range(4)}
    correct_label = next(label for label, text in options.items() if text == correct)
    return options, correct_label


def _build_case_bank_rows(
    candidates: list[StudyCandidate],
    preserved_rows: list[dict[str, Any]],
    *,
    case_target: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in preserved_rows]
    seen_ids = {str(row.get("case_id") or "") for row in rows}
    seen_titles = {str(row.get("title") or "") for row in rows}
    for candidate in candidates:
        if len(rows) >= case_target:
            break
        case_id = _case_record_id(candidate)
        title = _case_title(candidate)
        if case_id in seen_ids:
            continue
        if title in seen_titles:
            title = f"{title}（{candidate.sample_id[-6:]}）"
        rows.append(
            {
                "case_id": case_id,
                "title": title,
                "facts": _build_case_facts(candidate),
                "issue": _build_case_issue(candidate),
                "holding": _build_case_holding(candidate),
                "statutes": candidate.references[:3],
                "tags": candidate.tags,
            }
        )
        seen_ids.add(case_id)
        seen_titles.add(title)
    return rows


def _case_title(candidate: StudyCandidate) -> str:
    base = _compact_case_title_text(candidate.question)
    return f"{candidate.topic}案例：{base}"


def _compact_case_title_text(text: str, *, max_chars: int = 36) -> str:
    cleaned = clean_text(text).replace("？", "").replace("?", "").strip()
    if not cleaned:
        return "未命名案例"
    if len(cleaned) <= max_chars:
        return cleaned
    summarized = _summarize_complete_text(cleaned, max_chars=max_chars, max_sentences=1, allow_clause_split=False)
    if summarized:
        return summarized
    for marker in ("。", "，", ",", "：", ":"):
        if marker in cleaned:
            candidate = clean_text(cleaned.split(marker, maxsplit=1)[0])
            if candidate and len(candidate) <= max_chars:
                return candidate
    return cleaned[:max_chars].rstrip("，,；;：:")


def _build_case_facts(candidate: StudyCandidate) -> str:
    compact = _summarize_complete_text(candidate.question, max_chars=900, max_sentences=3, allow_clause_split=False)
    return compact or clean_text(candidate.question)


def _build_case_issue(candidate: StudyCandidate) -> str:
    question = clean_text(candidate.question)
    summarized = _summarize_complete_text(question, max_chars=220, max_sentences=1, allow_clause_split=False)
    return summarized or question


def _build_case_holding(candidate: StudyCandidate) -> str:
    compact = _summarize_complete_text(candidate.analysis or candidate.answer, max_chars=1200, max_sentences=4, allow_clause_split=False)
    return compact or clean_text(candidate.analysis or candidate.answer)


def _question_record_id(candidate: StudyCandidate) -> str:
    seed = f"{candidate.sample_id}|{candidate.question}|{candidate.answer_summary}"
    return f"auto-q-{_stable_score(seed):016x}"[:23]


def _case_record_id(candidate: StudyCandidate) -> str:
    seed = f"{candidate.sample_id}|{candidate.question}|{candidate.answer_summary}"
    return f"auto-case-{_stable_score(seed):016x}"[:26]


def _build_common_knowledge_rows(
    question_candidates: list[StudyCandidate],
    case_candidates: list[StudyCandidate],
    catalog_rows: list[dict[str, str]],
    preserved_rows: list[dict[str, Any]],
    *,
    common_target: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in preserved_rows]
    seen_ids = {str(row.get("entry_id") or "") for row in rows}
    seen_titles = {str(row.get("title") or "") for row in rows}

    for row in STATIC_COMMON_KNOWLEDGE:
        if len(rows) >= common_target:
            break
        if row["entry_id"] in seen_ids or row["title"] in seen_titles:
            continue
        rows.append(dict(row))
        seen_ids.add(row["entry_id"])
        seen_titles.add(row["title"])

    available_topics = {candidate.topic for candidate in question_candidates + case_candidates if candidate.topic in TOPIC_COMMON_GUIDES}
    for topic in TOPIC_ORDER:
        if topic not in available_topics:
            continue
        if len(rows) >= common_target:
            break
        guide = TOPIC_COMMON_GUIDES[topic]
        focus_entry = {
            "entry_id": f"auto-common-topic-focus-{topic}",
            "title": f"{topic}复习抓手",
            "content": f"复习{topic}时，优先围绕 { '、'.join(guide['keywords']) } 三类高频主题整理法条、构成要件和典型题型，再回到制度适用条件做判断。",
            "tags": [topic, "学习方法", "复习抓手"],
            "source": "auto_builder",
        }
        if focus_entry["entry_id"] not in seen_ids and focus_entry["title"] not in seen_titles:
            rows.append(focus_entry)
            seen_ids.add(focus_entry["entry_id"])
            seen_titles.add(focus_entry["title"])
        if len(rows) >= common_target:
            break
        pitfall_entry = {
            "entry_id": f"auto-common-topic-pitfall-{topic}",
            "title": f"{topic}常见失分点",
            "content": guide["pitfall"],
            "tags": [topic, "失分点", "学习方法"],
            "source": "auto_builder",
        }
        if pitfall_entry["entry_id"] not in seen_ids and pitfall_entry["title"] not in seen_titles:
            rows.append(pitfall_entry)
            seen_ids.add(pitfall_entry["entry_id"])
            seen_titles.add(pitfall_entry["title"])

    for row in GLOBAL_COMMON_GUIDES:
        if len(rows) >= common_target:
            break
        if row["entry_id"] in seen_ids or row["title"] in seen_titles:
            continue
        rows.append(dict(row))
        seen_ids.add(row["entry_id"])
        seen_titles.add(row["title"])

    for topic in TOPIC_ORDER:
        if len(rows) >= common_target:
            break
        if topic not in TOPIC_COMMON_GUIDES:
            continue
        guide = TOPIC_COMMON_GUIDES[topic]
        entry = {
            "entry_id": f"auto-common-topic-checklist-{topic}",
            "title": f"{topic}答题检查单",
            "content": f"回答{topic}题目前，至少检查主体是谁、争点在哪、适用条件是否满足，以及 { '、'.join(guide['keywords'][:2]) } 是否已经写到位。",
            "tags": [topic, "答题检查", "学习方法"],
            "source": "auto_builder",
        }
        if entry["entry_id"] in seen_ids or entry["title"] in seen_titles:
            continue
        rows.append(entry)
        seen_ids.add(entry["entry_id"])
        seen_titles.add(entry["title"])
    return rows[:common_target]


def _build_system_seed_rows(preserved_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in preserved_rows]
    seen_ids = {str(row.get("id") or "") for row in rows}
    for row in STATIC_SYSTEM_MEMORIES:
        if row["id"] in seen_ids:
            continue
        rows.append(dict(row))
        seen_ids.add(row["id"])
    return rows