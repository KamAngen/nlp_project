from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_agent.config import apply_configured_cuda_visible_devices, load_app_config


def cmd_build_corpus(args: argparse.Namespace) -> None:
    from legal_agent.data.corpus_builder import build_law_corpus

    config = load_app_config(args.config)
    summary = build_law_corpus(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_index(args: argparse.Namespace) -> None:
    config = load_app_config(args.config)
    if str(args.embedding_device).startswith("cuda"):
        apply_configured_cuda_visible_devices(config)

    from legal_agent.rag.index_builder import build_rag_index

    summary = build_rag_index(config, device=args.embedding_device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_download_disc_law(args: argparse.Namespace) -> None:
    from legal_agent.data.disc_law import download_disc_law_dataset, normalize_disc_law_dataset

    config = load_app_config(args.config)
    downloaded = download_disc_law_dataset(config.disc_law_raw_dir)
    normalized = normalize_disc_law_dataset(config.disc_law_raw_dir, config.disc_law_normalized_path)
    print(json.dumps({"downloaded_files": [str(path) for path in downloaded], "normalized_records": len(normalized)}, ensure_ascii=False, indent=2))


def cmd_prepare_law_data(args: argparse.Namespace) -> None:
    from legal_agent.data.law_dataset import prepare_law_dataset

    config = load_app_config(args.config)
    summary = prepare_law_dataset(
        args.source_root or (config.project_root / "law_files"),
        args.target_root or config.law_dir,
        cleanup_source=bool(args.cleanup_source),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_repair_law_docs(args: argparse.Namespace) -> None:
    from legal_agent.data.word_conversion import repair_law_documents

    config = load_app_config(args.config)
    summary = repair_law_documents(args.law_dir or config.law_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_data(args: argparse.Namespace) -> None:
    config = load_app_config(args.config)
    apply_configured_cuda_visible_devices(config)

    from legal_agent.training.dataset_builder import build_agent_datasets

    summary = build_agent_datasets(
        config,
        study_config_path=args.study_config,
        train_count=args.train_count,
        eval_count=args.eval_count,
        retrieval_device=args.retrieval_device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    config = load_app_config(args.config)
    apply_configured_cuda_visible_devices(config)

    from legal_agent.training.train_qlora import train_agent_qlora

    metrics = train_agent_qlora(
        config,
        train_path=args.train_path or config.generated_train_path,
        eval_path=args.eval_path or config.generated_eval_path,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_chat(args: argparse.Namespace) -> None:
    config = load_app_config(args.config)
    apply_configured_cuda_visible_devices(config)

    from legal_agent.study_agent import LegalStudyAgent
    from legal_agent.study_config import load_study_agent_config

    study_config = load_study_agent_config(args.study_config)
    agent = LegalStudyAgent(study_config, retrieval_device=args.retrieval_device)

    print("输入 exit 退出。")
    while True:
        question = input("\n[USER] ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        result = agent.handle_message(
            question,
            user_id=args.user_id,
            session_id=args.session_id,
            model_path=args.model_path or str(config.models.agent_base),
            adapter_path=args.adapter_path,
            prompt_mode=args.prompt_mode,
            retrieval_device=args.retrieval_device,
            model_device=args.model_device,
        )
        print("\n[ANSWER]")
        print(result.answer)


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_app_config(args.config)
    apply_configured_cuda_visible_devices(config)

    from legal_agent.evaluation.run_eval import evaluate_model

    payload = evaluate_model(
        config,
        dataset_path=args.dataset_path or config.generated_eval_path,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        retrieval_device=args.retrieval_device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_export_case_studies(args: argparse.Namespace) -> None:
    from legal_agent.evaluation.case_study import export_case_studies

    path = export_case_studies(args.base_results, args.adapted_results, args.output, max_examples=args.max_examples)
    print(str(path))


def cmd_web_ui(args: argparse.Namespace) -> None:
    config = load_app_config(args.config)
    apply_configured_cuda_visible_devices(config)

    from legal_agent.web.app import launch_web_ui

    launch_web_ui(
        config_path=args.config,
        study_config_path=args.study_config,
        host=args.host,
        port=args.port,
        retrieval_device=args.retrieval_device,
        public=config.public,
    )


def cmd_build_study_kb(args: argparse.Namespace) -> None:
    from legal_agent.config import load_app_config
    from legal_agent.data.study_knowledge import prepare_study_knowledge_assets
    from legal_agent.study_config import load_study_agent_config
    from rag_engine.builder import build_study_knowledge_assets

    app_config = load_app_config(args.app_config)
    study_config = load_study_agent_config(args.config)
    generation_summary = prepare_study_knowledge_assets(
        app_config,
        study_config,
        question_count=args.question_count,
        case_count=args.case_count,
        common_count=args.common_count,
        force_rebuild=bool(args.force_rebuild),
        auto_download_disc_law=bool(args.auto_download_disc_law),
    )
    manifest_summary = build_study_knowledge_assets(
        question_bank_path=study_config.question_bank_path,
        case_bank_path=study_config.case_bank_path,
        common_knowledge_path=study_config.common_knowledge_path,
        manifest_path=study_config.study_manifest_path,
        use_legacy_statute_rag=study_config.use_legacy_statute_rag,
        legacy_config_path=study_config.legacy_config_path,
        legacy_device=args.retrieval_device,
    )
    print(json.dumps({"study_assets": generation_summary, "manifest": manifest_summary}, ensure_ascii=False, indent=2))


def cmd_prepare_study_data(args: argparse.Namespace) -> None:
    from legal_agent.config import load_app_config
    from legal_agent.data.study_knowledge import prepare_study_knowledge_assets
    from legal_agent.study_config import load_study_agent_config

    app_config = load_app_config(args.config)
    study_config = load_study_agent_config(args.study_config)
    summary = prepare_study_knowledge_assets(
        app_config,
        study_config,
        question_count=args.question_count,
        case_count=args.case_count,
        common_count=args.common_count,
        force_rebuild=bool(args.force_rebuild),
        auto_download_disc_law=bool(args.auto_download_disc_law),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_study_ask(args: argparse.Namespace) -> None:
    from legal_agent.study_agent import LegalStudyAgent
    from legal_agent.study_config import load_study_agent_config

    config = load_study_agent_config(args.config)
    agent = LegalStudyAgent(config, retrieval_device=args.retrieval_device)
    result = agent.handle_message(
        args.question,
        user_id=args.user_id,
        session_id=args.session_id,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        prompt_mode=args.prompt_mode,
        retrieval_device=args.retrieval_device,
        model_device=args.model_device,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_study_chat(args: argparse.Namespace) -> None:
    from legal_agent.study_agent import LegalStudyAgent
    from legal_agent.study_config import load_study_agent_config

    config = load_study_agent_config(args.config)
    agent = LegalStudyAgent(config, retrieval_device=args.retrieval_device)
    print("输入 exit 退出。")
    while True:
        question = input("\n[USER] ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        result = agent.handle_message(
            question,
            user_id=args.user_id,
            session_id=args.session_id,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            prompt_mode=args.prompt_mode,
            retrieval_device=args.retrieval_device,
            model_device=args.model_device,
        )
        print("\n[ANSWER]")
        print(result.answer)
        if result.report_path:
            print(f"\n[REPORT] {result.report_path}")


def cmd_study_report(args: argparse.Namespace) -> None:
    from legal_agent.study_agent import LegalStudyAgent
    from legal_agent.study_config import load_study_agent_config

    config = load_study_agent_config(args.config)
    agent = LegalStudyAgent(config, retrieval_device=args.retrieval_device)
    payload = agent.generate_report_response(
        user_id=args.user_id,
        session_id=args.session_id,
        report_type=args.report_type,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        prompt_mode=args.prompt_mode,
        retrieval_device=args.retrieval_device,
        model_device=args.model_device,
    )
    print(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chinese legal agent project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_corpus = subparsers.add_parser("build-corpus")
    build_corpus.add_argument("--config", default="configs/defaults.yaml")
    build_corpus.set_defaults(func=cmd_build_corpus)

    build_index = subparsers.add_parser("build-index")
    build_index.add_argument("--config", default="configs/defaults.yaml")
    build_index.add_argument("--embedding-device", default="cpu")
    build_index.set_defaults(func=cmd_build_index)

    download = subparsers.add_parser("download-disc-law")
    download.add_argument("--config", default="configs/defaults.yaml")
    download.set_defaults(func=cmd_download_disc_law)

    prepare_law = subparsers.add_parser("prepare-law-data")
    prepare_law.add_argument("--config", default="configs/defaults.yaml")
    prepare_law.add_argument("--source-root", default=None)
    prepare_law.add_argument("--target-root", default=None)
    prepare_law.add_argument("--cleanup-source", action="store_true")
    prepare_law.set_defaults(func=cmd_prepare_law_data)

    repair_law = subparsers.add_parser("repair-law-docs")
    repair_law.add_argument("--config", default="configs/defaults.yaml")
    repair_law.add_argument("--law-dir", default=None)
    repair_law.set_defaults(func=cmd_repair_law_docs)

    build_data = subparsers.add_parser("build-data")
    build_data.add_argument("--config", default="configs/defaults.yaml")
    build_data.add_argument("--study-config", default="configs/study_agent.yaml")
    build_data.add_argument("--train-count", type=int, default=None)
    build_data.add_argument("--eval-count", type=int, default=None)
    build_data.add_argument("--retrieval-device", default="cpu")
    build_data.set_defaults(func=cmd_build_data)

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/defaults.yaml")
    train.add_argument("--train-path", default=None)
    train.add_argument("--eval-path", default=None)
    train.add_argument("--base-model", default=None)
    train.add_argument("--output-dir", default=None)
    train.set_defaults(func=cmd_train)

    chat = subparsers.add_parser("chat")
    chat.add_argument("--config", default="configs/defaults.yaml")
    chat.add_argument("--study-config", default="configs/study_agent.yaml")
    chat.add_argument("--model-path", default=None)
    chat.add_argument("--adapter-path", default=None)
    chat.add_argument("--user-id", default="demo_user")
    chat.add_argument("--session-id", default="demo_session")
    chat.add_argument("--model-device", default="auto")
    chat.add_argument("--retrieval-device", default="cpu")
    chat.add_argument("--prompt-mode", choices=["pure", "one_shot", "few_shot"], default="pure")
    chat.set_defaults(func=cmd_chat)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/defaults.yaml")
    evaluate.add_argument("--dataset-path", default=None)
    evaluate.add_argument("--model-path", default=None)
    evaluate.add_argument("--adapter-path", default=None)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--retrieval-device", default="cpu")
    evaluate.set_defaults(func=cmd_evaluate)

    export = subparsers.add_parser("export-case-studies")
    export.add_argument("--base-results", required=True)
    export.add_argument("--adapted-results", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--max-examples", type=int, default=3)
    export.set_defaults(func=cmd_export_case_studies)

    web_ui = subparsers.add_parser("web-ui")
    web_ui.add_argument("--config", default="configs/defaults.yaml")
    web_ui.add_argument("--study-config", default="configs/study_agent.yaml")
    web_ui.add_argument("--host", default="127.0.0.1")
    web_ui.add_argument("--port", type=int, default=7860)
    web_ui.add_argument("--retrieval-device", default="cpu")
    web_ui.set_defaults(func=cmd_web_ui)

    build_study_kb = subparsers.add_parser("build-study-kb")
    build_study_kb.add_argument("--config", default="configs/study_agent.yaml")
    build_study_kb.add_argument("--app-config", default="configs/defaults.yaml")
    build_study_kb.add_argument("--retrieval-device", default="cpu")
    build_study_kb.add_argument("--question-count", type=int, default=0, help="0 表示生成全部可用题目")
    build_study_kb.add_argument("--case-count", type=int, default=0, help="0 表示生成全部可用案例")
    build_study_kb.add_argument("--common-count", type=int, default=24)
    build_study_kb.add_argument("--force-rebuild", action="store_true")
    build_study_kb.add_argument("--auto-download-disc-law", action="store_true")
    build_study_kb.set_defaults(func=cmd_build_study_kb)

    prepare_study_data = subparsers.add_parser("prepare-study-data")
    prepare_study_data.add_argument("--config", default="configs/defaults.yaml")
    prepare_study_data.add_argument("--study-config", default="configs/study_agent.yaml")
    prepare_study_data.add_argument("--question-count", type=int, default=0, help="0 表示生成全部可用题目")
    prepare_study_data.add_argument("--case-count", type=int, default=0, help="0 表示生成全部可用案例")
    prepare_study_data.add_argument("--common-count", type=int, default=24)
    prepare_study_data.add_argument("--force-rebuild", action="store_true")
    prepare_study_data.add_argument("--auto-download-disc-law", action="store_true")
    prepare_study_data.set_defaults(func=cmd_prepare_study_data)

    study_ask = subparsers.add_parser("study-ask")
    study_ask.add_argument("--config", default="configs/study_agent.yaml")
    study_ask.add_argument("--question", required=True)
    study_ask.add_argument("--user-id", default="demo_user")
    study_ask.add_argument("--session-id", default="demo_session")
    study_ask.add_argument("--model-path", default=None)
    study_ask.add_argument("--adapter-path", default=None)
    study_ask.add_argument("--model-device", default="auto")
    study_ask.add_argument("--retrieval-device", default="cpu")
    study_ask.add_argument("--prompt-mode", choices=["pure", "one_shot", "few_shot"], default="pure")
    study_ask.set_defaults(func=cmd_study_ask)

    study_chat = subparsers.add_parser("study-chat")
    study_chat.add_argument("--config", default="configs/study_agent.yaml")
    study_chat.add_argument("--user-id", default="demo_user")
    study_chat.add_argument("--session-id", default="demo_session")
    study_chat.add_argument("--model-path", default=None)
    study_chat.add_argument("--adapter-path", default=None)
    study_chat.add_argument("--model-device", default="auto")
    study_chat.add_argument("--retrieval-device", default="cpu")
    study_chat.add_argument("--prompt-mode", choices=["pure", "one_shot", "few_shot"], default="pure")
    study_chat.set_defaults(func=cmd_study_chat)

    study_report = subparsers.add_parser("study-report")
    study_report.add_argument("--config", default="configs/study_agent.yaml")
    study_report.add_argument("--user-id", default="demo_user")
    study_report.add_argument("--session-id", default="demo_session")
    study_report.add_argument("--report-type", default="study_progress")
    study_report.add_argument("--model-path", default=None)
    study_report.add_argument("--adapter-path", default=None)
    study_report.add_argument("--model-device", default="auto")
    study_report.add_argument("--retrieval-device", default="cpu")
    study_report.add_argument("--prompt-mode", choices=["pure", "one_shot", "few_shot"], default="pure")
    study_report.set_defaults(func=cmd_study_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
