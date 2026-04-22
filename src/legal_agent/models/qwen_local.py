from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Thread
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


@dataclass(slots=True)
class ModelOutput:
    raw_text: str
    content: str
    reasoning: str


def _parse_reasoning(raw_text: str) -> tuple[str, str]:
    if "</think>" in raw_text:
        head, tail = raw_text.rsplit("</think>", maxsplit=1)
        reasoning = head.replace("<think>", "").strip()
        content = tail.strip()
        return reasoning, content
    return "", raw_text.strip()


class LocalQwenChatModel:
    def __init__(
        self,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        device_map: str | dict[str, Any] = "auto",
        load_in_4bit: bool = False,
        compute_dtype: str = "bfloat16",
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.adapter_path = Path(adapter_path).resolve() if adapter_path else None
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.compute_dtype = compute_dtype
        self.tokenizer = None
        self.model = None

    def _read_raw_config(self) -> dict[str, Any]:
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            return {}
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_transformers_config(self) -> tuple[Any, bool]:
        raw_config = self._read_raw_config()
        quantization = raw_config.get("quantization_config") or {}
        is_fp8_model = quantization.get("quant_method") == "fp8"
        if not is_fp8_model:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            return config, False

        sanitized = dict(raw_config)
        sanitized.pop("quantization_config", None)
        model_type = str(sanitized.pop("model_type"))
        config = AutoConfig.for_model(model_type, **sanitized)
        config.name_or_path = str(self.model_path)
        return config, True

    def _render_prompt(self, messages: list[dict[str, str]], enable_thinking: bool | None) -> str:
        assert self.tokenizer is not None
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        try:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _prepare_inputs(self, prompt: str) -> dict[str, Any]:
        assert self.tokenizer is not None
        assert self.model is not None
        inputs = self.tokenizer([prompt], return_tensors="pt")
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def _build_generation_kwargs(
        self,
        inputs: dict[str, Any],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        streamer: TextIteratorStreamer | None = None,
    ) -> dict[str, Any]:
        assert self.tokenizer is not None
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": presence_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = max(temperature, 1e-6)
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k
        if streamer is not None:
            generation_kwargs["streamer"] = streamer
        return generation_kwargs

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        use_cpu = self.device_map == "cpu"
        if use_cpu:
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if self.compute_dtype == "bfloat16" else torch.float16
        config, is_fp8_model = self._load_transformers_config()
        quantization_config = None
        if self.load_in_4bit and not is_fp8_model and not use_cpu:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            device_map=self.device_map,
            dtype=dtype,
            quantization_config=quantization_config,
        )
        self.model.eval()

        if self.adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path, local_files_only=True)
            self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 20,
        presence_penalty: float = 1.0,
        enable_thinking: bool | None = None,
    ) -> ModelOutput:
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        prompt = self._render_prompt(messages, enable_thinking)
        inputs = self._prepare_inputs(prompt)
        generation_kwargs = self._build_generation_kwargs(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
        )

        with torch.inference_mode():
            generated = self.model.generate(**generation_kwargs)

        output_tokens = generated[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        reasoning, content = _parse_reasoning(raw_text)
        return ModelOutput(raw_text=raw_text, content=content, reasoning=reasoning)

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 20,
        presence_penalty: float = 1.0,
        enable_thinking: bool | None = None,
    ):
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        prompt = self._render_prompt(messages, enable_thinking)
        inputs = self._prepare_inputs(prompt)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = self._build_generation_kwargs(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            streamer=streamer,
        )

        error_holder: list[BaseException] = []

        def _worker() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                error_holder.append(exc)

        worker = Thread(target=_worker, daemon=True)
        worker.start()

        pieces: list[str] = []
        for piece in streamer:
            pieces.append(piece)
            raw_text = "".join(pieces).strip()
            reasoning, content = _parse_reasoning(raw_text)
            yield ModelOutput(raw_text=raw_text, content=content, reasoning=reasoning)

        worker.join()
        if error_holder:
            raise error_holder[0]
