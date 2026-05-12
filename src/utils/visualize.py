"""
토마토 파이프라인 시각화 유틸리티.

핵심 인터페이스
--------------
BBoxSpec          : 바운딩박스 하나의 그리기 명세 (xyxy, lines, color, thickness)
draw_bboxes()     : BBoxSpec 리스트를 이미지에 일괄 렌더링
score_to_bgr()    : harvest score(0-30) → BGR 색상 그라데이션

오버레이 빌더
-------------
ripeness_to_specs()  : ripeness_results → BBoxSpec 리스트
harvest_to_specs()   : harvest_results  → BBoxSpec 리스트
build_ripeness_overlay() : 전체 탐지 + 익음도 색상 bbox 이미지 반환
build_harvest_overlay()  : ripen 토마토만 + score 색상 bbox 이미지 반환

저장·표시
---------
save_result_images() : ripeness·harvest 오버레이 이미지를 2장 저장 (cv2.imwrite)
visualize()          : matplotlib 2단 그리드 표시 (ripeness overlay + harvest ranking)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 기본 색상 (BGR) ───────────────────────────────────────────────────────
COLOR_RIPEN   = (0,   0, 220)    # 빨강
COLOR_UNRIPEN = (255, 255, 255)  # 흰색
COLOR_ERROR   = (128, 128, 128)  # 회색
COLOR_HARVEST = (0,  230,  50)   # 라임  (selected)


# ── 핵심 데이터 클래스 ────────────────────────────────────────────────────

@dataclass
class BBoxSpec:
    """바운딩박스 하나의 그리기 명세.

    Args:
        xyxy     : (x1, y1, x2, y2) 픽셀 좌표
        lines    : 표시할 텍스트 줄 목록 (여러 줄 가능)
        color    : BGR 색상 튜플
        thickness: 박스 선 두께 (기본 2)
    """
    xyxy: tuple[int, int, int, int]
    lines: list[str]
    color: tuple[int, int, int]
    thickness: int = 2


# ── 핵심 드로잉 함수 ──────────────────────────────────────────────────────

def draw_bboxes(
    img_bgr: np.ndarray,
    specs: list[BBoxSpec],
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """BBoxSpec 리스트를 이미지에 그립니다.

    각 spec의 (xyxy, lines, color, thickness)를 사용해
    박스 + 검정 배경 노란 텍스트 레이블을 렌더링합니다.

    Args:
        img_bgr      : 원본 BGR 이미지 (변경되지 않음)
        specs        : BBoxSpec 리스트
        font_scale   : OpenCV 텍스트 폰트 크기
        font_thickness: 텍스트 두께

    Returns:
        박스가 그려진 BGR ndarray (원본의 복사본)
    """
    vis = img_bgr.copy()
    font        = cv2.FONT_HERSHEY_SIMPLEX
    txt_color   = (0, 255, 255)  # 노란색
    bg_color    = (0, 0, 0)      # 검정
    bg_alpha    = 0.55           # 텍스트 배경 불투명도 (낮을수록 더 투명)

    for spec in specs:
        x1, y1, x2, y2 = spec.xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), spec.color, spec.thickness)

        # 라벨: 여러 줄을 bbox 위쪽에 아래→위 순으로 쌓음
        for j, txt in enumerate(spec.lines):
            (tw, th), bl = cv2.getTextSize(txt, font, font_scale, font_thickness)
            line_h = th + bl + 4
            ty     = max(y1 - 4 - j * line_h, th + 4)

            # 반투명 검정 배경
            bx1, by1 = x1,          ty - th - bl - 2
            bx2, by2 = x1 + tw + 4, ty + bl + 1
            overlay = vis.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
            cv2.addWeighted(overlay, bg_alpha, vis, 1 - bg_alpha, 0, vis)

            # 노란 텍스트
            cv2.putText(
                vis, txt, (x1 + 2, ty),
                font, font_scale, txt_color, font_thickness, cv2.LINE_AA,
            )

    return vis


# ── 색상 유틸 ─────────────────────────────────────────────────────────────

def score_to_bgr(
    score: int,
    max_score: int = 30,
) -> tuple[int, int, int]:
    """harvest score → BGR 그라데이션 색상.

    score가 낮을수록 파랑, 높을수록 빨강.

    Args:
        score    : 점수 (0 ~ max_score)
        max_score: 최대 점수 (기본 30)

    Returns:
        BGR 튜플
    """
    t = max(0.0, min(float(score), float(max_score))) / max_score
    return (int(255 * (1 - t)), 60, int(255 * t))


# ── BBoxSpec 변환 ─────────────────────────────────────────────────────────

def ripeness_to_specs(ripeness_results: list[tuple]) -> list[BBoxSpec]:
    """ripeness_results 를 BBoxSpec 리스트로 변환합니다.

    Args:
        ripeness_results: step2_ripeness() 반환값
            [(bbox, det_conf, label, reasoning), ...]

    Returns:
        BBoxSpec 리스트
    """
    specs = []
    print(ripeness_results)
    for i, ((cx1, cy1, cx2, cy2), conf, lbl, _) in enumerate(ripeness_results):
        color = (
            COLOR_RIPEN   if lbl == "ripe"
            else COLOR_UNRIPEN if lbl == "unripe"
            else COLOR_ERROR
        )
        thickness = 2 if lbl in ("ripe", "unripe") else 1
        specs.append(BBoxSpec(
            xyxy=(cx1, cy1, cx2, cy2),
            lines=[f"#{i + 1}  {conf:.2f}"],
            color=color,
            thickness=thickness,
        ))
    return specs


def harvest_to_specs(harvest_results: list[tuple]) -> list[BBoxSpec]:
    """harvest_results 를 BBoxSpec 리스트로 변환합니다.

    Args:
        harvest_results: step3_harvest() 반환값 (score 내림차순)
            [(orig_idx, bbox, det_conf, HarvestResult|None), ...]

    Returns:
        BBoxSpec 리스트
    """
    specs = []
    for rank, (orig_idx, (cx1, cy1, cx2, cy2), conf, hr) in enumerate(harvest_results, 1):
        if hr is None:
            specs.append(BBoxSpec(
                xyxy=(cx1, cy1, cx2, cy2),
                lines=[f"#{orig_idx + 1} ERROR"],
                color=COLOR_ERROR,
                thickness=3,
            ))
        else:
            specs.append(BBoxSpec(
                xyxy=(cx1, cy1, cx2, cy2),
                lines=[
                    f"#{orig_idx + 1}  score={hr.harvest_confidence_score}/30",
                    f"P={hr.reasoning.proximity_score}  "
                    f"A={hr.reasoning.access_score}  "
                    f"I={hr.reasoning.isolation_score}",
                ],
                color=COLOR_HARVEST,
                thickness=3,
            ))
    return specs


# ── 오버레이 빌더 ─────────────────────────────────────────────────────────

def build_ripeness_overlay(
    img_bgr: np.ndarray,
    ripeness_results: list[tuple],
) -> np.ndarray:
    """전체 탐지 토마토를 익음도 색상 bbox로 오버레이한 이미지를 반환합니다.

    ripen → 빨강  /  unripen → 초록  /  error → 회색

    Returns:
        BGR ndarray
    """
    return draw_bboxes(img_bgr, ripeness_to_specs(ripeness_results))


def build_harvest_overlay(
    img_bgr: np.ndarray,
    harvest_results: list[tuple],
) -> np.ndarray:
    """ripen 토마토만 harvest score 색상 bbox로 오버레이한 이미지를 반환합니다.

    score 낮을수록 파랑, 높을수록 빨강 (0-30 스케일).

    Returns:
        BGR ndarray
    """
    return draw_bboxes(img_bgr, harvest_to_specs(harvest_results))


# ── 저장 ─────────────────────────────────────────────────────────────────

def save_result_images(
    img_bgr: np.ndarray,
    ripeness_results: list[tuple],
    harvest_results: list[tuple],
    save_dir: Path,
) -> tuple[Path, Path, Path]:
    """ripeness·harvest 오버레이 이미지를 각각 저장합니다.

    저장 파일::

        <save_dir>/ripeness.jpg       ← 전체 탐지 + 익음도 색상 bbox
        <save_dir>/harvest_score.jpg  ← ripen만 + harvest score 색상 bbox
        <save_dir>/final_harvest.jpg  ← ripen만 + harvest score 색상 bbox
    Args:
        img_bgr         : 원본 BGR 이미지
        ripeness_results: step2_ripeness() 반환값
        harvest_results : step3_harvest() 반환값 (score 내림차순)
        save_dir        : 저장 디렉토리 (없으면 생성)

    Returns:
        (ripeness_path, harvest_score_path, final_harvest_path)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rip_path = save_dir / "ripeness.jpg"
    cv2.imwrite(str(rip_path), build_ripeness_overlay(img_bgr, ripeness_results))

    hvs_path = save_dir / "harvest_score.jpg"
    cv2.imwrite(str(hvs_path), build_harvest_overlay(img_bgr, harvest_results))

    # ── 1등 토마토만 bbox 표시 ─────────────────────────────────
    final_harvest_path = save_dir / "final_harvest.jpg"
    if harvest_results:
        _, (cx1, cy1, cx2, cy2), _, hr = harvest_results[0]
        color = score_to_bgr(hr.harvest_confidence_score) if hr is not None else COLOR_ERROR
        final_harvest_spec = BBoxSpec(xyxy=(cx1, cy1, cx2, cy2), lines=[], color=color, thickness=3)
        cv2.imwrite(str(final_harvest_path), draw_bboxes(img_bgr, [final_harvest_spec]))
    else:
        cv2.imwrite(str(final_harvest_path), img_bgr.copy())

    return rip_path, hvs_path, final_harvest_path


# ── matplotlib 표시 ───────────────────────────────────────────────────────

def visualize(
    img_bgr: np.ndarray,
    ripeness_results: list[tuple],
    harvest_results: list[tuple],
    max_cols: int = 3,
) -> None:
    """파이프라인 결과를 matplotlib 2단 그리드로 화면에 표시합니다.

    상단 행: ripeness 오버레이 전체 이미지
    하단 행: harvest 랭킹 (토마토별 full image + 하이라이트 bbox)

    Args:
        img_bgr         : 원본 BGR 이미지
        ripeness_results: step2_ripeness() 반환값
        harvest_results : step3_harvest() 반환값 (score 내림차순)
        max_cols        : harvest 그리드 최대 열 수
    """
    vis_rip = build_ripeness_overlay(img_bgr, ripeness_results)
    n_ripen = sum(1 for _, _, lbl, _ in ripeness_results if lbl == "ripen")

    n_h = len(harvest_results)
    n_cols = min(n_h, max_cols) if n_h > 0 else 1
    n_rows_h = math.ceil(n_h / n_cols) if n_h > 0 else 0
    total_rows = 1 + n_rows_h

    fig = plt.figure(figsize=(max(14, n_cols * 7), 7 + n_rows_h * 5))

    # ── 상단: ripeness overlay ────────────────────────────────
    ax_top = fig.add_subplot(total_rows, 1, 1)
    ax_top.imshow(vis_rip[..., ::-1])
    ax_top.axis("off")
    ax_top.set_title(
        f"YOLO + Ripeness  ({len(ripeness_results)} detected  |  {n_ripen} ripen)  "
        "■ red=ripen  ■ green=unripen",
        fontsize=13, fontweight="bold",
    )

    # ── 하단: harvest ranking grid ────────────────────────────
    if harvest_results:
        hvs_specs = harvest_to_specs(harvest_results)
        img_rgb = img_bgr[..., ::-1]

        for rank, ((orig_idx, (cx1, cy1, cx2, cy2), conf, hr), spec) in enumerate(
            zip(harvest_results, hvs_specs)
        ):
            row = rank // n_cols + 1
            col = rank % n_cols
            ax = fig.add_subplot(total_rows, n_cols, n_cols * row + col + 1)

            # 해당 토마토만 하이라이트한 전체 이미지
            vis_bgr = draw_bboxes(img_bgr, [spec])
            ax.imshow(vis_bgr[..., ::-1])
            ax.axis("off")

            if hr is not None:
                score = hr.harvest_confidence_score
                t = score / 30.0
                title = (
                    f"Rank {rank + 1}  |  #{orig_idx + 1}  score={score}/30\n"
                    f"P={hr.reasoning.proximity_score}  "
                    f"A={hr.reasoning.access_score}  "
                    f"I={hr.reasoning.isolation_score}  "
                    f"det={conf:.2f}"
                )
                c = (t, 0.1, 1 - t)
            else:
                title = f"Rank {rank + 1}  |  #{orig_idx + 1}  ERROR"
                c = (0.5, 0.5, 0.5)

            ax.set_title(title, fontsize=10, fontweight="bold", color=c, pad=4)
            ax.text(
                0.03, 0.97, f"#{rank + 1}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0),
            )

        fig.suptitle(
            f"Harvest Ranking  ({n_h} ripen tomatoes, sorted by score)  "
            "■ blue=low  ■ red=high",
            fontsize=14, y=1.01,
        )

    plt.tight_layout()
    plt.show()
