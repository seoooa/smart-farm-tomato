"""Tomato harvest pipeline.

Flow:
    Step 1  YOLO detection      → 이미지에서 토마토 bbox 탐지
    Step 2  Ripeness (Qwen3.5)  → 각 bbox를 crop하여 ripe / unripe 분류
    Step 3  Harvest  (Qwen3.5)  → ripe 토마토만 모아 전체 이미지에서 최적 수확 대상 선정

Usage:
    python script/pipeline.py --input <image_or_dir> [options]
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
from pathlib import Path

import autorootcwd
import click
import cv2
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
from ultralytics import YOLO

import src.models.yolo_v26_sft_detection.inference as yolo_infer
import src.models.qwen3_5_sft_ripeness.inference_ripeness as ripeness_infer
import src.models.qwen3_5_sft_harvest.inference_harvest as harvest_infer
from src.models.qwen3_5_sft_ripeness.inference_ripeness import RipenessResult
from src.models.qwen3_5_sft_harvest.inference_harvest import HarvestResult
from src.utils.visualize import BBoxSpec, draw_bboxes

_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_YOLO_WEIGHT    = str(_ROOT / "src" / "models" / "yolo_v26_sft_detection" / "weights" / "yolo_v26s_twoclass.pt")
_DEFAULT_RIPENESS_MODEL = str(_ROOT / "src" / "models" / "qwen3_5_sft_ripeness" / "weights" / "VLLM_float16" / "qwen3.5_0.8b_lora")
_DEFAULT_HARVEST_MODEL  = str(_ROOT / "src" / "models" / "qwen3_5_sft_harvest" / "weights" / "VLLM_float16" / "qwen3.5_0.8b_lora")
_DEFAULT_CONF    = 0.7
_DEFAULT_IOU     = 0.45
_DEFAULT_DEVICE  = 3
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _crop_bbox(
    img_bgr: np.ndarray,
    xyxy: list[float],
    pad: int = 4,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """이미지에서 bbox crop (패딩 포함)."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(xyxy[0]) - pad)
    y1 = max(0, int(xyxy[1]) - pad)
    x2 = min(w, int(xyxy[2]) + pad)
    y2 = min(h, int(xyxy[3]) + pad)
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


# ── Step 1: YOLO detection ────────────────────────────────────────────────────

def _merge_to_tomato(detections: list) -> list:
    """multi-class YOLO 결과를 클래스 구분 없이 단일 'tomato' 클래스로 통합합니다.

    두 클래스(예: ripe/unripe) 모두 ripeness 모듈에서 재분류할 것이므로
    YOLO 클래스 레이블은 무시하고 bbox·confidence만 유지합니다.
    """
    for det in detections:
        det.class_id   = 0
        det.class_name = "tomato"
    return detections


def step1_detect(
    image_path: Path,
    yolo_model: YOLO,
    conf: float,
    iou: float,
    device: int,
) -> tuple[np.ndarray, list]:
    """YOLO로 토마토를 탐지합니다.

    multi-class 모델이더라도 모든 탐지 결과를 단일 'tomato' 클래스로
    통합한 뒤 반환합니다. 클래스 재분류는 Step 2 Ripeness 모듈이 담당합니다.

    Returns:
        (img_bgr, detections):
            img_bgr    : 원본 BGR numpy 이미지
            detections : Detection 객체 리스트 (class_name='tomato'으로 통합)
    """
    raw_results = yolo_model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        save=False,
    )
    detections = yolo_infer.parse_results(raw_results, yolo_model.names)
    detections = _merge_to_tomato(detections)
    img_bgr = cv2.imread(str(image_path))
    return img_bgr, detections


# ── Step 2: Ripeness classification ──────────────────────────────────────────

def step2_ripeness(
    img_bgr: np.ndarray,
    detections: list,
    model,
    tokenizer,
    crop_pad: int = 4,
    batch_size: int = 8,
) -> list[tuple[tuple[int, int, int, int], float, RipenessResult | None]]:
    """탐지된 모든 bbox를 crop한 뒤 배치로 익음도를 분류합니다.

    순차 처리 대신 batch_size 단위로 묶어 한 번에 추론하므로
    토마토 수가 많을수록 속도 이점이 커집니다.

    Returns:
        [(bbox, det_conf, RipenessResult | None), ...]
            bbox     : (x1, y1, x2, y2) 절댓값 픽셀
            det_conf : YOLO detection confidence
    """
    crops, bboxes, confs = [], [], []
    for det in detections:
        crop_bgr, bbox = _crop_bbox(img_bgr, det.xyxy, crop_pad)
        crops.append(PILImage.fromarray(crop_bgr[..., ::-1]))
        bboxes.append(bbox)
        confs.append(float(det.confidence))

    ripeness_list = ripeness_infer.predict_batch(crops, model, tokenizer, batch_size=batch_size)
    return list(zip(bboxes, confs, ripeness_list))


# ── Step 3: Harvest selection ─────────────────────────────────────────────────

def step3_harvest(
    img_bgr: np.ndarray,
    ripeness_results: list[tuple[tuple[int, int, int, int], float, RipenessResult | None]],
    model,
    tokenizer,
) -> HarvestResult | None:
    """ripe 토마토만 골라 전체 이미지에서 최적 수확 대상을 선정합니다.

    Returns:
        HarvestResult, 또는 ripe 후보 없음 / 파싱 실패 시 None.
    """
    ripe_tomatoes = [
        {"id": i + 1, "bbox": list(bbox)}
        for i, (bbox, _conf, result) in enumerate(ripeness_results)
        if result is not None and result.label == "ripe"
    ]
    if not ripe_tomatoes:
        return None

    img_pil = PILImage.fromarray(img_bgr[..., ::-1])
    return harvest_infer.predict(img_pil, ripe_tomatoes, model, tokenizer)


# ── 결과 시각화 ───────────────────────────────────────────────────────────────

# 3-tier 색상 (BGR) — visualize.py COLOR_* 와 통일
_COLOR_UNRIPE   = (255, 255, 255)  # 흰색  — YOLO 탐지, unripe
_COLOR_RIPE     = (0,     0, 220)  # 빨강  — ripe 판정
_COLOR_SELECTED = (0,   230,  50)  # 라임  — 최종 수확 선택

def save_pipeline_image(
    img_bgr: np.ndarray,
    ripeness_results: list[tuple[tuple[int, int, int, int], float, RipenessResult | None]],
    harvest_result: HarvestResult | None,
    save_dir: Path,
) -> Path:
    """파이프라인 결과를 3색 bbox로 시각화하여 저장합니다.

    색상 기준:
        회색 (unripe/unknown) : YOLO 탐지됐으나 ripe 아님
        주황 (ripe)           : Ripeness 모델이 ripe로 판정 + 수확 점수 표시
        라임 (selected)       : 최종 수확 선택 토마토

    Returns:
        저장된 이미지 경로 (<save_dir>/pipeline.jpg).
    """
    selected_id = harvest_result.selected_tomato_id if harvest_result else -1

    # id(1-based) → total_score 맵
    score_map: dict[int, int] = {}
    if harvest_result:
        for s in harvest_result.tomato_scores:
            score_map[s.id] = s.total_score

    specs: list[BBoxSpec] = []
    for i, (bbox, det_conf, result) in enumerate(ripeness_results):
        tomato_id = i + 1
        label     = result.label if result is not None else "unknown"

        if tomato_id == selected_id:
            score  = score_map.get(tomato_id)
            color  = _COLOR_SELECTED
            lines  = [
                f"#{tomato_id} SELECTED",
                f"score:{score}" if score is not None else "",
            ]
            thickness = 3
        elif label == "ripe":
            score  = score_map.get(tomato_id)
            color  = _COLOR_RIPE
            lines  = [
                f"#{tomato_id} ripe",
                f"score:{score}" if score is not None else "",
            ]
            thickness = 2
        else:
            color     = _COLOR_UNRIPE
            lines     = [f"#{tomato_id} {label}"]
            thickness = 2

        specs.append(BBoxSpec(xyxy=bbox, lines=[l for l in lines if l], color=color, thickness=thickness))

    vis = draw_bboxes(img_bgr, specs)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "pipeline.jpg"
    cv2.imwrite(str(out_path), vis)
    return out_path


# ── 결과 저장 ─────────────────────────────────────────────────────────────────

def save_results_json(
    ripeness_results: list[tuple],
    harvest_result: HarvestResult | None,
    save_dir: Path,
) -> Path:
    """파이프라인 결과를 JSON으로 저장합니다.

    Returns:
        저장된 파일 경로 (<save_dir>/results.json).
    """
    ripeness_out = []
    for i, (bbox, conf, result) in enumerate(ripeness_results):
        x1, y1, x2, y2 = bbox
        entry: dict = {
            "id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "det_conf": round(conf, 4),
        }
        if result is not None:
            entry["label"]     = result.label
            entry["is_ripe"]   = result.is_ripe
            entry["reasoning"] = result.reasoning
        else:
            entry["label"]     = "parse_error"
            entry["is_ripe"]   = None
            entry["reasoning"] = None
        ripeness_out.append(entry)

    # selected_tomato_id → bbox 역조회 맵
    id_to_bbox = {entry["id"]: entry["bbox"] for entry in ripeness_out}

    harvest_out: dict | None = None
    if harvest_result is not None:
        selected_id   = harvest_result.selected_tomato_id
        selected_bbox = id_to_bbox.get(selected_id)
        harvest_out = {
            "selected_tomato_id": selected_id,
            "selected_bbox": selected_bbox,
            "reasoning": harvest_result.reasoning,
            "tomato_scores": [
                {
                    "id": s.id,
                    "ripeness_score":   s.ripeness_score,
                    "visibility_score": s.visibility_score,
                    "isolation_score":  s.isolation_score,
                    "total_score":      s.total_score,
                }
                for s in harvest_result.tomato_scores
            ],
        }

    data = {
        "harvest": harvest_out,
        "ripeness": ripeness_out,
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "results.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ── 단일 이미지 파이프라인 ────────────────────────────────────────────────────

def run_single(
    image: Path,
    yolo_model: YOLO,
    conf: float,
    iou: float,
    device: int,
    ripeness_model,
    ripeness_tokenizer,
    harvest_model,
    harvest_tokenizer,
) -> None:
    """이미지 1장에 대해 파이프라인 전체를 실행하고 결과를 저장합니다."""
    img_bgr, detections = step1_detect(image, yolo_model, conf, iou, device)
    if not detections:
        tqdm.write(f"[SKIP] {image.name}: 탐지된 토마토 없음")
        return

    ripeness_results = step2_ripeness(img_bgr, detections, ripeness_model, ripeness_tokenizer)
    harvest_result   = step3_harvest(img_bgr, ripeness_results, harvest_model, harvest_tokenizer)

    save_dir = _ROOT / "result" / image.stem
    save_results_json(ripeness_results, harvest_result, save_dir)
    save_pipeline_image(img_bgr, ripeness_results, harvest_result, save_dir)
    tqdm.write(
        f"[OK] {image.name}  "
        f"detect={len(detections)}  "
        f"ripe={sum(1 for _, _, r in ripeness_results if r and r.label == 'ripe')}  "
        f"harvest_id={harvest_result.selected_tomato_id if harvest_result else 'N/A'}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input", "input_path", required=True,
    type=click.Path(exists=True, path_type=Path),
    help="입력 이미지 파일 또는 이미지가 담긴 폴더 경로",
)
@click.option(
    "--yolo-weight", default=_DEFAULT_YOLO_WEIGHT, show_default=True,
    help="YOLO 가중치 파일 경로 (.pt)",
)
@click.option(
    "--ripeness-model", "ripeness_model_path", default=_DEFAULT_RIPENESS_MODEL, show_default=True,
    help="Ripeness 분류 모델 경로 (unsloth 저장 형식)",
)
@click.option(
    "--harvest-model", "harvest_model_path", default=_DEFAULT_HARVEST_MODEL, show_default=True,
    help="Harvest 선택 모델 경로 (unsloth 저장 형식)",
)
@click.option("--conf",   default=_DEFAULT_CONF,   show_default=True, help="YOLO confidence 임계값")
@click.option("--iou",    default=_DEFAULT_IOU,    show_default=True, help="YOLO NMS IoU 임계값")
@click.option("--device", default=_DEFAULT_DEVICE, type=int, show_default=True, help="YOLO GPU 번호")
@click.option(
    "--load-in-4bit", is_flag=True, default=False, show_default=True,
    help="Qwen 모델을 4-bit 양자화로 로드합니다",
)
def main(
    input_path: Path,
    yolo_weight: str,
    ripeness_model_path: str,
    harvest_model_path: str,
    conf: float,
    iou: float,
    device: int,
    load_in_4bit: bool,
):
    """이미지(또는 폴더)에 대해 YOLO 탐지 → Ripeness 분류 → Harvest 선택 파이프라인을 실행합니다."""

    # ── 입력 목록 수집 ────────────────────────────────────────────────────────
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in _IMG_EXTS)
    else:
        images = [input_path]

    if not images:
        click.echo("처리할 이미지가 없습니다.")
        return

    # ── 모델 로드 (전체 공유, 각 1회) ────────────────────────────────────────
    tqdm.write(f"[1/3] YOLO 로드: {yolo_weight}")
    yolo_model = YOLO(yolo_weight)

    tqdm.write(f"[2/3] Ripeness 모델 로드: {ripeness_model_path}")
    ripeness_model, ripeness_tokenizer = ripeness_infer.load_model(
        ripeness_model_path, load_in_4bit=load_in_4bit
    )

    tqdm.write(f"[3/3] Harvest 모델 로드: {harvest_model_path}")
    harvest_model, harvest_tokenizer = harvest_infer.load_model(
        harvest_model_path, load_in_4bit=load_in_4bit
    )
    tqdm.write("모델 로드 완료.\n")

    # ── 이미지별 파이프라인 실행 ──────────────────────────────────────────────
    errors: list[str] = []
    with tqdm(images, desc="Pipeline", unit="img") as pbar:
        for image in pbar:
            pbar.set_postfix(file=image.name[:30])
            try:
                run_single(
                    image=image,
                    yolo_model=yolo_model,
                    conf=conf,
                    iou=iou,
                    device=device,
                    ripeness_model=ripeness_model,
                    ripeness_tokenizer=ripeness_tokenizer,
                    harvest_model=harvest_model,
                    harvest_tokenizer=harvest_tokenizer,
                )
            except Exception as e:
                errors.append(image.name)
                tqdm.write(f"[ERROR] {image.name}: {e}")

    tqdm.write(f"\n완료: {len(images)}개 처리  /  오류: {len(errors)}개")
    if errors:
        tqdm.write("오류 발생 파일: " + ", ".join(errors))


if __name__ == "__main__":
    main()
