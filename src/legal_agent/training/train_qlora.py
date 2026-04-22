from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from legal_agent.config import AppConfig, load_app_config
from legal_agent.utils.io import ensure_dir, read_jsonl


def _render_prompt(example: dict[str, Any], tokenizer: AutoTokenizer, enable_thinking: bool = False) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(example["messages"][:-1], **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(example["messages"][:-1], **kwargs)


def _render_prompt_completion(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    return {
        "prompt": _render_prompt(example, tokenizer, enable_thinking=False),
        "completion": example["messages"][-1]["content"],
    }


@dataclass
class PromptCompletionCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_id_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []

        for feature in features:
            prompt_ids = self.tokenizer(feature["prompt"], add_special_tokens=False).input_ids
            completion_ids = self.tokenizer(feature["completion"], add_special_tokens=False).input_ids
            if not completion_ids or completion_ids[-1] != self.tokenizer.eos_token_id:
                completion_ids = completion_ids + [self.tokenizer.eos_token_id]

            input_ids = prompt_ids + completion_ids
            labels = ([-100] * len(prompt_ids)) + completion_ids
            if len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length :]
                labels = labels[-self.max_length :]
            attention_mask = [1] * len(input_ids)

            input_id_rows.append(input_ids)
            label_rows.append(labels)
            attention_rows.append(attention_mask)

        batch_max = max(len(row) for row in input_id_rows)
        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        for input_ids, labels, attention_mask in zip(input_id_rows, label_rows, attention_rows):
            pad_len = batch_max - len(input_ids)
            input_ids.extend([pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(input_id_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }


def train_agent_qlora(
    config: AppConfig,
    *,
    train_path: str | Path,
    eval_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    base_model_path = Path(base_model_path or config.models.agent_base).resolve()
    output_dir = Path(output_dir or config.training.output_dir).resolve()
    ensure_dir(output_dir)

    if torch.cuda.is_available():
        training_device_map: dict[str, int | str] = {"": torch.cuda.current_device()}
    else:
        training_device_map = {"": "cpu"}

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = read_jsonl(train_path)
    eval_records = read_jsonl(eval_path) if eval_path else []
    train_dataset = Dataset.from_list([_render_prompt_completion(record, tokenizer) for record in train_records])
    eval_dataset = Dataset.from_list([_render_prompt_completion(record, tokenizer) for record in eval_records]) if eval_records else None

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map=training_device_map,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.training.target_modules,
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=4,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        label_names=["labels"],
        logging_dir=str(output_dir / "logs"),
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
    )
    collator = PromptCompletionCollator(tokenizer=tokenizer, max_length=config.training.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "loss_type": "completion_only_cross_entropy",
        "output_dir": config.project_relative_path(output_dir),
        "base_model": config.project_relative_path(base_model_path),
        "best_checkpoint": config.project_relative_path(trainer.state.best_model_checkpoint),
    }
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA training for the legal agent model.")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_app_config(args.config)
    train_path = args.train_path or config.generated_train_path
    eval_path = args.eval_path or (config.generated_eval_path if Path(config.generated_eval_path).exists() else None)
    metrics = train_agent_qlora(
        config,
        train_path=train_path,
        eval_path=eval_path,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
