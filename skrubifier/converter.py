"""
Stage 2: Converter. LLM-agnostic — pass any `llm_call(prompt: str) -> str`
callable (e.g. wrapping the Anthropic SDK). Kept decoupled from a specific
SDK so this framework can run under whatever credentials/model the user has
configured, and so tests can inject a fake LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .ir import PipelineIR
from .prompts import build_conversion_prompt, REPAIR_TEMPLATE

LLMCall = Callable[[str], str]


def _extract_code(response: str) -> str:
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fall back: assume the whole response is code if no fence found
    return response.strip()


def default_anthropic_llm_call(model: str = "claude-sonnet-4-6") -> LLMCall:
    """Convenience factory. Requires `pip install anthropic` and
    ANTHROPIC_API_KEY set in the environment. Not used automatically — the
    caller wires this in explicitly (see cli.py)."""
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return call


def default_openai_compatible_llm_call(
    model: str = "qwen3-coder-next",
    base_url: str = "https://chat-ai.academiccloud.de/v1",
    api_key_env: str = "GWDG_API_KEY",
) -> LLMCall:
    """Convenience factory for any OpenAI-compatible endpoint, defaulting to
    GWDG/AcademicCloud's SAIA service (free, open-weight models hosted in
    Germany — see https://kisski.gwdg.de). Requires `pip install openai` and
    the relevant API key set in the environment (default: GWDG_API_KEY).

    Model choice: prefer a model whose docs/benchmarks emphasize coding
    ability, since the converter's task is code generation, not chat. Check
    currently available models with:
        curl https://chat-ai.academiccloud.de/v1/models \\
             -H "Authorization: Bearer $GWDG_API_KEY"
    Good starting candidates as of writing: qwen3-coder-next,
    openai-gpt-oss-120b, devstral-2-123b-instruct-2512 — swap the `model`
    argument if a given one isn't available/ready when you check.
    """
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    return call


@dataclass
class ConversionResult:
    code: str
    attempts: int
    validation_report: Optional[dict] = None
    success: bool = False


def convert(
    ir: PipelineIR,
    original_source: str,
    llm_call: LLMCall,
    validate_fn: Optional[Callable[[str], dict]] = None,
    max_repair_rounds: int = 3,
) -> ConversionResult:
    """
    validate_fn(code) -> {"ok": bool, "detail": str, ...}  (see validator.py)
    If validate_fn is None, only a single LLM pass is done (no repair loop) —
    useful when running outside a sandbox with the target libraries.
    """
    prompt = build_conversion_prompt(json.dumps(ir.to_dict(), indent=2, default=str), original_source)
    response = llm_call(prompt)
    code = _extract_code(response)

    if validate_fn is None:
        return ConversionResult(code=code, attempts=1)

    for attempt in range(1, max_repair_rounds + 1):
        report = validate_fn(code)
        if report.get("ok"):
            return ConversionResult(code=code, attempts=attempt, validation_report=report, success=True)
        repair_prompt = REPAIR_TEMPLATE.format(failure=report.get("detail", "unknown failure"), candidate=code)
        response = llm_call(repair_prompt)
        code = _extract_code(response)

    final_report = validate_fn(code)
    return ConversionResult(code=code, attempts=max_repair_rounds + 1,
                             validation_report=final_report, success=final_report.get("ok", False))
