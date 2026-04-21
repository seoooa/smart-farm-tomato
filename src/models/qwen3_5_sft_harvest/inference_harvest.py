"""Harvest selection inference using Qwen3.5-0.8B (unsloth).

Pipeline position:
    YOLO detection → Ripeness classification → **Harvest selection** ← here

Public API
----------
load_model(model_path, load_in_4bit) → (model, tokenizer)
run_inference(pil_image, ripe_tomatoes, model, tokenizer) → raw_str
parse_output(raw_str) → dict | None
predict(pil_image, ripe_tomatoes, model, tokenizer) → HarvestResult | None
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """\
You are a tomato harvest-selection annotation assistan.

Your task is to evaluate all ripe tomato candidates in the image and generate one output JSON object.
You must assign scores to every tomato listed in ripe_tomatoes and always select the single best tomato for harvest.

For each candidate, assign:
- ripeness_score (0-10): redness and maturity
- visibility_score (0-10): visibility considering leaves, stems, blur, and image border
- isolation_score (0-10): separation from other tomatoes

Rules:
- Compare tomatoes RELATIVE to each other within the same image.
- Set total_score = ripeness_score + visibility_score + isolation_score.
- Always select exactly one candidate ID from ripe_tomatoes.
"""

USER_PROMPT = """\
Choose the best harvest tomato from ripe_tomatoes.

Input:
{{
  "image_size": {image_size},
  "ripe_tomatoes": {ripe_tomatoes}
}}

Example JSON output:
{{
  "selected_tomato_id": 2,
  "reasoning": "Tomato 2 is more uniformly red and clearly isolated than the other ripe candidates."
  "tomato_scores": [
    {{
      "id": 1,
      "ripeness_score": 7,
      "visibility_score": 8,
      "isolation_score": 5,
      "total_score": 20
    }},
    {{
      "id": 2,
      "ripeness_score": 9,
      "visibility_score": 8,
      "isolation_score": 7,
      "total_score": 24
    }}
  ]
}}

Rules:
- Score every tomato in ripe_tomatoes.
- total_score must equal ripeness_score + visibility_score + isolation_score.
- tomato_scores must contain one score object for every tomato in ripe_tomatoes.
- reasoning must be exactly ONE short sentence to explain the selected tomato using comparative evidence based on the scores.
- Output only the JSON object, nothing else.
"""

# max_new_tokens 계산 상수
_BASE_TOK = 100
_PER_TOMATO_TOK = 50


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class TomatoScore:
    id: int
    ripeness_score: int
    visibility_score: int
    isolation_score: int
    total_score: int


@dataclass
class HarvestResult:
    selected_tomato_id: int
    reasoning: str
    tomato_scores: list[TomatoScore] = field(default_factory=list)
    raw: str = ""


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model(model_path: str, load_in_4bit: bool = False):
    """Unsloth FastVisionModel을 로드하고 추론 모드로 설정합니다.

    Args:
        model_path: HuggingFace 모델 ID 또는 로컬 저장 경로.
        load_in_4bit: True면 4-bit 양자화로 로드합니다.

    Returns:
        (model, tokenizer) 튜플.
    """
    from unsloth import FastVisionModel  # 지연 임포트 (GPU 환경에서만 사용)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path, load_in_4bit=load_in_4bit
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


# ── 추론 ──────────────────────────────────────────────────────────────────────

def run_inference(
    pil_image: Image.Image,
    ripe_tomatoes: list[dict[str, Any]],
    model,
    tokenizer,
) -> str:
    """모델에 이미지와 ripe_tomatoes를 입력하고 raw 문자열을 반환합니다.

    Args:
        pil_image: 원본 PIL 이미지.
        ripe_tomatoes: [{"id": int, "bbox": [x1, y1, x2, y2]}, ...] 형태의 리스트.
        model: load_model()로 얻은 모델.
        tokenizer: load_model()로 얻은 토크나이저.

    Returns:
        모델이 생성한 raw 텍스트.
    """
    image_size = list(pil_image.size)  # [W, H]
    prompt_text = USER_PROMPT.format(
        image_size=json.dumps(image_size),
        ripe_tomatoes=json.dumps(ripe_tomatoes, ensure_ascii=False),
    )
    template_msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    input_text = tokenizer.apply_chat_template(
        template_msgs,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        pil_image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    max_new_tokens = _BASE_TOK + _PER_TOMATO_TOK * len(ripe_tomatoes)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def parse_output(raw: str) -> dict[str, Any] | None:
    """raw 문자열에서 JSON을 추출하고 파싱합니다.

    Returns:
        파싱된 dict, 또는 실패 시 None.
    """
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _dict_to_result(parsed: dict[str, Any], raw: str) -> HarvestResult:
    """파싱된 dict를 HarvestResult 데이터클래스로 변환합니다."""
    scores = [
        TomatoScore(
            id=s.get("id", -1),
            ripeness_score=s.get("ripeness_score", 0),
            visibility_score=s.get("visibility_score", 0),
            isolation_score=s.get("isolation_score", 0),
            total_score=s.get("total_score", 0),
        )
        for s in parsed.get("tomato_scores", [])
        if isinstance(s, dict)
    ]
    return HarvestResult(
        selected_tomato_id=parsed.get("selected_tomato_id", -1),
        reasoning=parsed.get("reasoning", ""),
        tomato_scores=scores,
        raw=raw,
    )


# ── 통합 API ──────────────────────────────────────────────────────────────────

def predict(
    pil_image: Image.Image,
    ripe_tomatoes: list[dict[str, Any]],
    model,
    tokenizer,
) -> HarvestResult | None:
    """이미지와 ripe_tomatoes를 받아 수확 선택 결과를 반환합니다.

    Args:
        pil_image: 원본 PIL 이미지.
        ripe_tomatoes: [{"id": int, "bbox": [x1, y1, x2, y2]}, ...].
        model: load_model()로 얻은 모델.
        tokenizer: load_model()로 얻은 토크나이저.

    Returns:
        HarvestResult, 또는 파싱 실패 시 None.
    """
    if not ripe_tomatoes:
        return None

    raw = run_inference(pil_image, ripe_tomatoes, model, tokenizer)
    parsed = parse_output(raw)
    if parsed is None:
        return None
    return _dict_to_result(parsed, raw)
