"""
Model backends.

Three of them, behind one `generate(question) -> sql` interface, because the
eval compares them head to head:

  FineTunedModel   The model this project trains. Sees the raw completion
                   format from prompt.build_prompt — the same text it saw
                   during SFT.

  BaseModel        The same base checkpoint, un-finetuned, prompted through its
                   chat template with the identical schema and business rules.
                   This is the honest baseline: it gets everything the
                   fine-tuned model gets, in the form it expects. Handicapping
                   the baseline is the easiest way to make a fine-tune look
                   good and the fastest way to make the result worthless.

  AnthropicModel   A frontier model over the API, same prompt. Answers the
                   question a reader will actually ask: is a 2B fine-tune worth
                   anything when a large model is one HTTP call away?

  GoldModel        Returns the reference SQL. Used to test the harness itself —
                   it must score 100%, and if it doesn't the bug is in the
                   comparison logic, not the model.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from prompt import build_chat_messages, build_prompt, extract_sql

DEFAULT_FINETUNED = os.getenv("FINETUNED_MODEL", "Hecodes/claims-text-to-sql")
DEFAULT_BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


class SQLModel(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str) -> str:
        """Return SQL for the question, or '' if none could be produced."""


class GoldModel(SQLModel):
    """Oracle backend: replays reference SQL. Sanity-checks the eval harness."""

    name = "gold (harness check)"

    def __init__(self, lookup: dict[str, str]):
        self._lookup = lookup

    def generate(self, question: str) -> str:
        return self._lookup.get(question, "")


class _HFModel(SQLModel):
    def __init__(self, model_id: str, max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.getenv("HF_TOKEN")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            token=token,
        )
        self._model.eval()

    def _decode(self, text: str) -> str:
        self._load()
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        completion = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return extract_sql(completion)


class FineTunedModel(_HFModel):
    """The fine-tune. Prompted exactly as it was trained — no chat template."""

    def __init__(self, model_id: str = DEFAULT_FINETUNED, **kwargs):
        super().__init__(model_id, **kwargs)
        self.name = f"fine-tuned ({model_id})"

    def generate(self, question: str) -> str:
        return self._decode(build_prompt(question))


class BaseModel(_HFModel):
    """The un-finetuned base checkpoint, prompted through its chat template."""

    def __init__(self, model_id: str = DEFAULT_BASE, **kwargs):
        super().__init__(model_id, **kwargs)
        self.name = f"base zero-shot ({model_id})"

    def generate(self, question: str) -> str:
        self._load()
        text = self._tokenizer.apply_chat_template(
            build_chat_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._decode(text)


class AnthropicModel(SQLModel):
    """Frontier baseline over the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"frontier zero-shot ({model})"
        self._client = None

    def generate(self, question: str) -> str:
        if self._client is None:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(api_key=api_key)

        messages = build_chat_messages(question)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=messages[0]["content"],
            messages=[{"role": "user", "content": messages[1]["content"]}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return extract_sql(text)


def build_model(spec: str, gold_lookup: dict[str, str] | None = None) -> SQLModel:
    """
    Resolve a CLI backend spec.

      gold                      oracle, for testing the harness
      finetuned[:model_id]      the fine-tune
      base[:model_id]           un-finetuned base checkpoint
      anthropic[:model_name]    frontier API baseline
    """
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()

    if kind == "gold":
        if gold_lookup is None:
            raise ValueError("gold backend requires the reference set")
        return GoldModel(gold_lookup)
    if kind == "finetuned":
        return FineTunedModel(arg or DEFAULT_FINETUNED)
    if kind == "base":
        return BaseModel(arg or DEFAULT_BASE)
    if kind == "anthropic":
        return AnthropicModel(arg or "claude-sonnet-4-5")
    raise ValueError(f"Unknown backend {spec!r}")
