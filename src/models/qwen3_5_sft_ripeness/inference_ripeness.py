"""Ripeness classification inference using Qwen3.5-0.8B (unsloth).

Pipeline position:
    YOLO detection → **Ripeness classification** ← here → Harvest selection

Public API
----------
load_model(model_path, load_in_4bit) → (model, tokenizer)
run_inference(pil_image, model, tokenizer) → raw_str
parse_output(raw_str) → dict | None
predict(pil_image, model, tokenizer) → RipenessResult | None
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """\
You are an expert tomato ripeness classifier.

Definitions:
- ripe: The center tomato shows any visible sign of ripening —
  fully red, predominantly red, or beginning to turn red
  (early blush, orange tint, reddish tint, or partial red coloration).
  It does NOT need to be fully red.

- unripe: The center tomato shows no sign of redness —
  it is fully green, yellow, or clearly pre-ripening.

Rules:
- Focus only on the CENTER tomato. Ignore borders, leaves, stems, and background.
- Base the decision on overall visible color, not tiny local artifacts.
"""

USER_PROMPT = """\
Classify the CENTER tomato as ripe or unripe.

Return exactly one valid JSON object:
{
  "is_tomato": 1,
  "is_ripe": 0,
  "reasoning": ""
}

Rules:
- "is_tomato": 1 if a tomato is visible, else 0.
- "is_ripe": 1 if ripe, else 0.
- "reasoning": one sentence describing the visual evidence.
- Output only the JSON object, nothing else.
"""

MAX_NEW_TOKENS = 96
CROP_SIZE = 512  # crop 이미지를 리사이즈할 크기 — 노트북과 동일


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class RipenessResult:
    is_tomato: int       # 1 = 토마토 있음, 0 = 없음
    is_ripe: int         # 1 = ripe, 0 = unripe
    reasoning: str
    raw: str = ""

    @property
    def label(self) -> str:
        """'ripe' | 'unripe' | 'not_tomato' 문자열 라벨."""
        if not self.is_tomato:
            return "not_tomato"
        return "ripe" if self.is_ripe else "unripe"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model(model_path: str, load_in_4bit: bool = False):
    """Unsloth FastVisionModel을 로드하고 추론 모드로 설정합니다.

    Args:
        model_path: HuggingFace 모델 ID 또는 로컬 저장 경로.
        load_in_4bit: True면 4-bit 양자화로 로드합니다.

    Returns:
        (model, tokenizer) 튜플.
    """
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path, load_in_4bit=load_in_4bit
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


# ── 추론 ──────────────────────────────────────────────────────────────────────

def run_inference(
    pil_image: Image.Image,
    model,
    tokenizer,
) -> str:
    """모델에 이미지를 입력하고 raw 문자열을 반환합니다.

    Args:
        pil_image: 토마토 crop 이미지 (PIL). CROP_SIZE로 리사이즈됩니다.
        model: load_model()로 얻은 모델.
        tokenizer: load_model()로 얻은 토크나이저.

    Returns:
        모델이 생성한 raw 텍스트.
    """
    img = (
        pil_image.resize((CROP_SIZE, CROP_SIZE))
        if pil_image.size != (CROP_SIZE, CROP_SIZE)
        else pil_image
    )
    template_msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    input_text = tokenizer.apply_chat_template(
        template_msgs,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        img,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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


# ── 통합 API ──────────────────────────────────────────────────────────────────

def _make_result(parsed: dict[str, Any] | None, raw: str) -> RipenessResult | None:
    if parsed is None:
        return None
    return RipenessResult(
        is_tomato=int(parsed.get("is_tomato", 1)),
        is_ripe=int(parsed.get("is_ripe", 0)),
        reasoning=parsed.get("reasoning", ""),
        raw=raw,
    )


def predict(
    pil_image: Image.Image,
    model,
    tokenizer,
) -> RipenessResult | None:
    """이미지를 받아 익음도 분류 결과를 반환합니다.

    Args:
        pil_image: 토마토 crop 이미지 (PIL).
        model: load_model()로 얻은 모델.
        tokenizer: load_model()로 얻은 토크나이저.

    Returns:
        RipenessResult, 또는 파싱 실패 시 None.
    """
    raw = run_inference(pil_image, model, tokenizer)
    return _make_result(parse_output(raw), raw)


def predict_batch(
    pil_images: list[Image.Image],
    model,
    tokenizer,
    batch_size: int = 8,
) -> list[RipenessResult | None]:
    """여러 crop 이미지를 배치로 묶어 익음도를 한 번에 분류합니다.

    단일 이미지에서 탐지된 토마토 여러 개를 순차 처리 대신 배치로
    처리하여 GPU 활용률을 높입니다.

    Args:
        pil_images : 토마토 crop 이미지 리스트 (PIL).
        model      : load_model()로 얻은 모델.
        tokenizer  : load_model()로 얻은 토크나이저.
        batch_size : 한 번에 처리할 이미지 수 (VRAM에 맞게 조정).

    Returns:
        입력 순서와 동일한 RipenessResult | None 리스트.
    """
    tokenizer.padding_side = "left"

    results: list[RipenessResult | None] = []

    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]

        # 리사이즈
        resized = [
            img.resize((CROP_SIZE, CROP_SIZE)) if img.size != (CROP_SIZE, CROP_SIZE) else img
            for img in batch
        ]

        # 각 이미지의 input text 생성 (동일한 프롬프트 반복)
        template_msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        texts = [
            tokenizer.apply_chat_template(
                template_msgs,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for _ in resized
        ]

        inputs = tokenizer(
            resized,
            texts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        for out in out_ids:
            raw = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            results.append(_make_result(parse_output(raw), raw))

    return results
