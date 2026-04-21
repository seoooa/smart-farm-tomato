from __future__ import annotations

import click
import cv2
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from ultralytics import YOLO


@dataclass
class Detection:
    """단일 객체 탐지 결과."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float] = field(default_factory=list)  # [x1, y1, x2, y2]

    def __str__(self) -> str:
        x1, y1, x2, y2 = self.xyxy
        w, h = x2 - x1, y2 - y1
        return (
            f"{self.class_name} conf={self.confidence:.3f} "
            f"bbox=[{x1:.1f}, {y1:.1f}, {w:.1f}×{h:.1f}]"
        )


def parse_results(results, names: dict[int, str]) -> list[Detection]:
    """ultralytics.predict() 결과를 Detection 리스트로 변환."""
    detections: list[Detection] = []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return detections

    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    clses = r0.boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clses):
        cls_id = int(cls_id)
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=names.get(cls_id, str(cls_id)),
                confidence=float(c),
                xyxy=[float(x1), float(y1), float(x2), float(y2)],
            )
        )
    return detections


def resolve_source(data: str) -> str:
    """데이터 경로를 이미지 소스 경로로 변환.

    - yaml 파일이면 val/images 경로를 파싱해서 반환
    - 디렉토리면 val/images → train/images → images 순으로 탐색
    - 그 외(이미지/파일)는 그대로 반환
    """
    p = Path(data)

    # yaml 파일인 경우
    if p.suffix in (".yaml", ".yml") and p.is_file():
        with open(p) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg.get("path", p.parent))
        for split in ("val", "train"):
            candidate = base / cfg.get(split, f"{split}/images")
            if candidate.exists():
                click.echo(f"[Predict] yaml에서 소스 경로 감지: {candidate}")
                return str(candidate)

    # 디렉토리인 경우 이미지 하위 폴더 탐색
    if p.is_dir():
        for subdir in ("val/images", "train/images", "images"):
            candidate = p / subdir
            if candidate.exists():
                click.echo(f"[Predict] 소스 경로 자동 감지: {candidate}")
                return str(candidate)

    return data


def predict(
    weight: str,
    data: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: int = 0,
    name: str | None = None,
    project: str | Path = "result/detection",
    save: bool = True,
    save_txt: bool = False,
) -> tuple:
    """모델 추론 (Inference).

    Returns:
        (raw_results, detections): ultralytics 원본 결과 리스트와
        Detection 리스트의 튜플.
    """
    weight = str(weight)
    data = str(data)

    weight_stem = Path(weight).stem
    data_stem = Path(data).stem
    name = name or data_stem

    project = Path(project)
    project = str((project / weight_stem).resolve())

    click.echo(f"[Predict] 가중치: {weight}")
    click.echo(f"[Predict] 데이터: {data}")
    click.echo(f"[Predict] conf={conf}, iou={iou}, device={device}, save={save}")
    click.echo(f"[Predict] 저장 경로: {project}/{name}")

    model = YOLO(weight)
    source = resolve_source(data)

    raw_results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        device=device,
        name=name,
        project=project,
        save=save,
        save_txt=save_txt,
    )

    detections = parse_results(raw_results, model.names)
    click.echo(f"[Predict] 추론 완료 → {project}/{name}  ({len(detections)}개 탐지)")
    return raw_results, detections


def predict_realtime(
    weight: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: int = 0,
    camera_idx: int = 0,
) -> None:
    """웹캠 실시간 토마토 탐지.

    ESC 또는 'q' 키로 종료합니다.

    Args:
        weight     : 가중치 파일 경로 (.pt)
        conf       : confidence 임계값
        iou        : NMS IoU 임계값
        device     : GPU 번호 (CPU: -1)
        camera_idx : 카메라 디바이스 인덱스 (기본값 0 = 첫 번째 웹캠)
    """
    click.echo(f"[Realtime] 가중치: {weight}")
    click.echo(f"[Realtime] 카메라: {camera_idx}  conf={conf}, iou={iou}, device={device}")
    click.echo("[Realtime] 종료하려면 ESC 또는 'q' 키를 누르세요.")

    yolo_model = YOLO(weight)
    names = yolo_model.names

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise click.ClickException(f"카메라 {camera_idx}를 열 수 없습니다.")

    # 박스·텍스트 스타일
    BOX_THICKNESS = 2
    FONT_SCALE = 0.55
    FONT_THICKNESS = 1
    BOX_COLOR = (0, 120, 255)   # BGR 주황
    TEXT_COLOR = (255, 255, 255)
    TEXT_BG_ALPHA = 0.55

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                click.echo("[Realtime] 프레임을 읽을 수 없습니다. 카메라를 확인하세요.")
                break

            results = yolo_model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )

            r0 = results[0]
            vis = frame.copy()

            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy  = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clses = r0.boxes.cls.cpu().numpy()

                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clses):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    label = f"{names.get(int(cls_id), str(int(cls_id)))} {c:.2f}"

                    # 박스
                    cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                    # 텍스트 배경
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                    )
                    ty1 = max(y1 - th - baseline - 4, 0)
                    overlay = vis.copy()
                    cv2.rectangle(
                        overlay,
                        (x1, ty1),
                        (x1 + tw + 4, ty1 + th + baseline + 4),
                        BOX_COLOR,
                        -1,
                    )
                    cv2.addWeighted(overlay, TEXT_BG_ALPHA, vis, 1 - TEXT_BG_ALPHA, 0, vis)

                    # 텍스트
                    cv2.putText(
                        vis,
                        label,
                        (x1 + 2, ty1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE,
                        TEXT_COLOR,
                        FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

            # FPS 표시
            fps_text = f"tomatoes: {len(r0.boxes) if r0.boxes is not None else 0}"
            cv2.putText(
                vis, fps_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow("Tomato Detection (ESC/q to quit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        click.echo("[Realtime] 종료.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--weight", required=True, help="추론에 사용할 가중치 파일 경로 (.pt)")
@click.option("--data", default=None, help="추론 대상 (이미지/디렉토리/영상). --realtime 사용 시 불필요.")
@click.option("--conf", default=0.25, show_default=True, help="신뢰도(confidence) 임계값")
@click.option("--iou", default=0.45, show_default=True, help="NMS IoU 임계값")
@click.option("--device", default=1, type=int, show_default=True, help="추론 디바이스 GPU 번호")
@click.option(
    "--name",
    default=None,
    show_default=True,
    help="결과 저장 폴더 이름 (기본값: 데이터 파일명/스텀)",
)
@click.option(
    "--project",
    default="result/detection",
    show_default=True,
    type=click.Path(path_type=Path),
    help="결과 저장 상위 디렉토리",
)
@click.option("--save/--no-save", default=True, show_default=True, help="탐지 결과 이미지 저장 여부")
@click.option("--save-txt/--no-save-txt", default=False, show_default=True, help="탐지 결과 txt 저장 여부")
@click.option("--realtime", is_flag=True, default=False, help="웹캠 실시간 탐지 모드 활성화")
@click.option("--camera", default=0, type=int, show_default=True, help="실시간 모드에서 사용할 카메라 디바이스 인덱스")
def cli(weight, data, conf, iou, device, name, project, save, save_txt, realtime, camera):
    """YOLOv26 추론(inference).

    실시간 모드: --realtime [--camera 0]
    파일/디렉토리 모드: --data <경로>
    """
    if realtime:
        predict_realtime(
            weight=weight,
            conf=conf,
            iou=iou,
            device=device,
            camera_idx=camera,
        )
        return

    if data is None:
        raise click.UsageError("--data 옵션이 필요합니다. 실시간 모드는 --realtime 플래그를 사용하세요.")

    _, detections = predict(
        weight=weight,
        data=data,
        conf=conf,
        iou=iou,
        device=device,
        name=name,
        project=project,
        save=save,
        save_txt=save_txt,
    )
    for det in detections:
        click.echo(f"  {det}")


if __name__ == "__main__":
    cli()

