"""Harvest score inference using Qwen3.5-0.8B (unsloth).

Pipeline position:
    YOLO detection → Ripeness classification → **Harvest score prediction** ← here

Public API
----------
load_model(model_path, load_in_4bit) → (model, tokenizer)
run_inference(pil_image, ripe_tomatoes, model, tokenizer) → list[str]
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
You are a tomato harvest score prediction assistant.

Your task is to score ONE target tomato using two images:
1. The first image is the full scene image.
2. The second image is a crop image of the target tomato.

Input metadata gives the full image size, target tomato id, and target_bbox in full-image coordinates.

Score definitions:
- ripeness_score (0-10): judge only color maturity and redness of the target tomato.
- visibility_score (0-10): judge only how clearly the target tomato itself can be seen. Penalize leaves, stems, blur, darkness, glare, and border truncation. Do not penalize nearby tomatoes unless they occlude the target tomato.
- isolation_score (0-10): judge only tomato-to-tomato separation. Penalize touching, overlap, clustering, and crowding by other tomatoes. Do not penalize leaves, stems, blur, or color here.

Evidence usage:
- Use the crop image mainly for ripeness and visibility.
- Use the full image mainly for isolation and scene context.
- Score only the target tomato, not the whole image.
- Assign each criterion independently. Do not let a high score in one criterion compensate for another criterion.

Score anchors:
- 0-2: poor
- 3-4: low
- 5-6: moderate
- 7-8: good
- 9-10: excellent

Return only valid JSON. No markdown, no extra text.
"""

USER_PROMPT = """\
Score the target tomato.

Input:
{{
  "image_size": {image_size},
  "target_tomato_id": {target_tomato_id},
  "target_bbox": {target_bbox}
}}

Return exactly one valid JSON object:
{{
  "id": {target_tomato_id},
  "ripeness_score": 0,
  "visibility_score": 0,
  "isolation_score": 0
}}

Requirements:
- The output id must equal target_tomato_id.
- Scores must be integers from 0 to 10.
- Do not output total_score, selected_tomato_id, or reasoning.
- Output only the JSON object and nothing else.
"""

MAX_NEW_TOKENS = 80
CROP_SIZE = 512
CROP_PADDING = 20


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class TomatoScore:
    id: int
    ripeness_score: int
    visibility_score: int
    isolation_score: int
    total_score: int
    bbox_area: int = 0
    raw: str = ""


@dataclass
class HarvestResult:
    selected_tomato_id: int
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


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _bbox_area(bbox: list[int | float]) -> int:
    try:
        x1, y1, x2, y2 = bbox
        return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
    except Exception:
        return 0


def _safe_int_score(value, default: int = 0) -> int:
    try:
        return max(0, min(10, int(value)))
    except Exception:
        return default


def _crop_candidate(
    pil_image: Image.Image,
    bbox: list[int | float],
) -> Image.Image:
    W, H = pil_image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - CROP_PADDING)
    y1 = max(0, int(y1) - CROP_PADDING)
    x2 = min(W, int(x2) + CROP_PADDING)
    y2 = min(H, int(y2) + CROP_PADDING)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1

    crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
    if crop.size != (CROP_SIZE, CROP_SIZE):
        crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
    return crop


# ── 추론 ──────────────────────────────────────────────────────────────────────

def run_inference(
    pil_image: Image.Image,
    ripe_tomatoes: list[dict[str, Any]],
    model,
    tokenizer,
) -> list[str]:
    """각 ripe tomato에 대해 full image + crop image를 배치로 넣고 raw 출력 리스트를 반환합니다."""
    if not ripe_tomatoes:
        return []

    full_img = pil_image.convert("RGB")
    image_size = list(full_img.size)
    all_images = []
    input_texts = []

    for tomato in ripe_tomatoes:
        crop = _crop_candidate(full_img, tomato["bbox"])
        prompt_text = USER_PROMPT.format(
            image_size=json.dumps(image_size),
            target_tomato_id=json.dumps(tomato["id"]),
            target_bbox=json.dumps(tomato["bbox"]),
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        input_texts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        all_images.extend([full_img, crop])

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        images=all_images,
        text=input_texts,
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
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        )

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in out_ids
    ]


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def parse_output(raw: str) -> dict[str, Any] | None:
    """raw 문자열에서 harvest score JSON을 추출하고 파싱합니다."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_harvest_score(
    raw: str,
    fallback_id: int,
    bbox: list[int | float],
) -> TomatoScore:
    parsed = parse_output(raw) or {}
    try:
        tomato_id = int(parsed.get("id", fallback_id))
    except Exception:
        tomato_id = fallback_id

    ripeness_score = _safe_int_score(parsed.get("ripeness_score"), 0)
    visibility_score = _safe_int_score(parsed.get("visibility_score"), 0)
    isolation_score = _safe_int_score(parsed.get("isolation_score"), 0)
    total_score = ripeness_score + visibility_score + isolation_score

    return TomatoScore(
        id=tomato_id,
        ripeness_score=ripeness_score,
        visibility_score=visibility_score,
        isolation_score=isolation_score,
        total_score=total_score,
        bbox_area=_bbox_area(bbox),
        raw=raw,
    )


# ── 통합 API ──────────────────────────────────────────────────────────────────

def predict(
    pil_image: Image.Image,
    ripe_tomatoes: list[dict[str, Any]],
    model,
    tokenizer,
) -> HarvestResult | None:
    """ripe 토마토별 점수를 예측하고, total_score 기준으로 최종 수확 대상을 반환합니다.

    total_score가 같으면 bbox 면적이 더 큰 토마토를 선택합니다.
    """
    if not ripe_tomatoes:
        return None

    raw_outputs = run_inference(pil_image, ripe_tomatoes, model, tokenizer)
    scores = [
        _parse_harvest_score(raw, fallback_id=tomato["id"], bbox=tomato["bbox"])
        for raw, tomato in zip(raw_outputs, ripe_tomatoes)
    ]
    if not scores:
        return None

    selected = max(scores, key=lambda s: (s.total_score, s.bbox_area))
    raw_summary = json.dumps(
        {"per_candidate_raw_outputs": {str(t["id"]): raw for t, raw in zip(ripe_tomatoes, raw_outputs)}},
        ensure_ascii=False,
    )
    return HarvestResult(
        selected_tomato_id=selected.id,
        tomato_scores=scores,
        raw=raw_summary,
    )
