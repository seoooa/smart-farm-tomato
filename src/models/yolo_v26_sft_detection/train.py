import sys
import click
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.visualize import BBoxSpec, draw_bboxes, COLOR_RIPEN, COLOR_UNRIPEN, COLOR_ERROR


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


@click.group()
def cli():
    """YOLOv26 토마토 탐지 모델 학습 및 추론"""
    pass


@cli.command()
@click.option(
    "--data",
    default="data/merged_tomato_yolo/data.yaml",
    show_default=True,
    help="데이터셋 yaml 파일 경로",
)
@click.option(
    "--weights",
    default="yolo26s.pt",
    show_default=True,
    help="초기 가중치 파일 경로 (사전학습 모델 또는 .pt 파일)",
)
@click.option("--max_epochs", default=100, show_default=True, help="학습 에폭 수")
@click.option(
    "--batch",
    default=32,
    show_default=True,
    help="배치 크기 (-1=AutoBatch)",
)
@click.option("--imgsz", default=640, show_default=True, help="입력 이미지 크기 (픽셀)")
@click.option("--device", default=0, type=int, show_default=True, help="학습 디바이스 GPU 번호 (0, 1, 2, ...)")
@click.option(
    "--name",
    default=None,
    show_default=True,
    help="결과 저장 폴더 이름 (기본값: 데이터 파일명)",
)
@click.option(
    "--project",
    default="result/detection",
    show_default=True,
    help="결과 저장 상위 디렉토리",
)
@click.option(
    "--optimizer",
    default="AdamW",
    show_default=True,
    type=click.Choice(["SGD", "Adam", "AdamW", "auto"], case_sensitive=False),
    help="옵티마이저 (auto 선택 시 lr 무시됨)",
)
@click.option("--lr", default=0.001, show_default=True, help="초기 학습률 (optimizer=auto 시 무시됨)")
@click.option("--patience", default=30, show_default=True, help="Early stopping patience")
@click.option(
    "--freeze",
    default=None,
    type=int,
    show_default=True,
    help="동결할 레이어 수 (소규모 데이터 fine-tuning 시 권장: 10)",
)
@click.option("--cos_lr/--no_cos_lr", default=True, show_default=True, help="Cosine LR 스케줄링 사용 여부")
def train(data, weights, max_epochs, batch, imgsz, device, name, project, optimizer, lr, patience, freeze, cos_lr):
    """모델 파인튜닝 (Fine-tuning)"""
    name = name or Path(data).stem

    # 데이터 yaml에서 nc를 읽어 single_class / two_class 서브디렉토리 자동 결정
    data_path = Path(data)
    if data_path.suffix in (".yaml", ".yml") and data_path.is_file():
        with open(data_path) as f:
            data_cfg = yaml.safe_load(f)
        nc = data_cfg.get("nc", 1)
        class_subdir = "two_class" if nc >= 2 else "single_class"
        project = str(Path(project) / class_subdir)
        click.echo(f"[Train] 클래스 수: {nc} → 저장 경로: {project}")

    click.echo(f"[Train] 가중치: {weights}")
    click.echo(f"[Train] 데이터셋: {data}")
    click.echo(f"[Train] max_epochs={max_epochs}, batch={batch}, imgsz={imgsz}, device={device}")
    click.echo(f"[Train] optimizer={optimizer}, lr={lr}, patience={patience}, freeze={freeze}, cos_lr={cos_lr}")

    if optimizer.lower() == "auto":
        click.echo("[Train] 경고: optimizer=auto 는 lr 설정을 무시합니다. AdamW 또는 SGD를 권장합니다.")

    model = YOLO(weights)

    train_kwargs = dict(
        data=data,
        epochs=max_epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        name=name,
        project=str(Path(project).resolve()),
        optimizer=optimizer,
        lr0=lr,
        patience=patience,
        cos_lr=cos_lr,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    results = model.train(**train_kwargs)

    click.echo(f"[Train] 학습 완료 → {project}/{name}")
    return results


@cli.command()
@click.option(
    "--weight",
    required=True,
    help="추론에 사용할 가중치 파일 경로 (.pt)",
)
@click.option(
    "--data",
    required=True,
    help="추론 대상 (이미지 파일, 디렉토리, 영상 파일, 0=웹캠)",
)
@click.option(
    "--conf",
    default=0.25,
    show_default=True,
    help="신뢰도(confidence) 임계값",
)
@click.option("--iou", default=0.45, show_default=True, help="NMS IoU 임계값")
@click.option("--device", default=0, type=int, show_default=True, help="추론 디바이스 GPU 번호 (0, 1, 2, ...)")
@click.option(
    "--name",
    default=None,
    show_default=True,
    help="결과 저장 폴더 이름 (기본값: 가중치 파일명)",
)
@click.option(
    "--project",
    default="result/260410",
    show_default=True,
    type=click.Path(path_type=Path),
    help="결과 저장 상위 디렉토리",
)
@click.option("--save/--no-save", default=True, show_default=True, help="결과 이미지 저장 여부")
@click.option("--save-txt/--no-save-txt", default=False, show_default=True, help="탐지 결과 txt 저장 여부")
def predict(weight, data, conf, iou, device, name, project, save, save_txt):
    """모델 추론 (Inference)"""
    weight_stem = Path(weight).stem
    data_stem = Path(data).stem
    name = name or data_stem
    project = str(Path(project / weight_stem).resolve())

    click.echo(f"[Predict] 가중치: {weight}")
    click.echo(f"[Predict] 데이터: {data}")
    click.echo(f"[Predict] conf={conf}, iou={iou}, device={device}, save={save}")
    click.echo(f"[Predict] 저장 경로: {project}/{name}")

    model = YOLO(weight)
    source = resolve_source(data)

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        device=device,
        name=name,
        project=project,
        save=save,
        save_txt=save_txt,
    )

    click.echo(f"[Predict] 추론 완료 → {project}/{name}")
    return results


@cli.command()
@click.option("--weight", required=True, help="평가할 가중치 파일 경로 (.pt)")
@click.option(
    "--data",
    default="data/merged_tomato_yolo/data.yaml",
    show_default=True,
    help="데이터셋 yaml 파일 경로",
)
@click.option(
    "--split",
    default="test",
    show_default=True,
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    help="평가할 데이터 분할",
)
@click.option("--conf", default=0.001, show_default=True, help="metrics 계산용 confidence 임계값")
@click.option("--vis_conf", default=0.25, show_default=True, help="시각화용 confidence 임계값")
@click.option("--iou", default=0.6, show_default=True, help="NMS IoU 임계값")
@click.option("--imgsz", default=640, show_default=True, help="입력 이미지 크기")
@click.option("--batch", default=32, show_default=True, help="배치 크기")
@click.option("--device", default=0, type=int, show_default=True, help="GPU 번호")
@click.option("--project", default="result/detection", show_default=True, help="결과 저장 상위 디렉토리")
@click.option("--name", default=None, show_default=True, help="결과 저장 폴더 이름")
def evaluate(weight, data, split, conf, vis_conf, iou, imgsz, batch, device, project, name):
    """테스트셋 성능 평가 (P / R / F1 / mAP50 / mAP50-95) + 커스텀 bbox 시각화 저장"""
    weight_stem = Path(weight).stem
    name = name or f"{weight_stem}_{split}_eval"

    # yaml에서 nc, split 이미지 경로 파싱
    data_path = Path(data)
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get("nc", 1)
    class_subdir = "two_class" if nc >= 2 else "single_class"
    project_dir = Path(project) / class_subdir
    save_dir = (Path(project_dir) / name).resolve()
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(data_cfg.get("path", data_path.parent))
    split_rel = data_cfg.get(split, f"{split}/images")
    img_dir = dataset_root / split_rel

    click.echo(f"[Evaluate] 가중치    : {weight}")
    click.echo(f"[Evaluate] 데이터셋  : {data}  (split={split})")
    click.echo(f"[Evaluate] 이미지 경로: {img_dir}")
    click.echo(f"[Evaluate] 결과 저장 : {save_dir}")

    model = YOLO(weight)
    names = model.names  # {0: 'ripe', 1: 'unripe'}

    # ── 1. 정량 지표 계산 (model.val) ────────────────────────────────────────
    click.echo("\n[Evaluate] 지표 계산 중...")
    metrics = model.val(
        data=data,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir.resolve()),
        name=name,
        plots=True,
        verbose=False,
    )

    box = metrics.box

    # 클래스별 P / R / F1 / mAP50 / mAP50-95
    CLASS_COLORS = {0: COLOR_RIPEN, 1: COLOR_UNRIPEN}
    rows = []
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            p  = float(box.p[i])
            r  = float(box.r[i])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append({
                "class": names.get(int(cls_idx), str(int(cls_idx))),
                "P": p, "R": r, "F1": f1,
                "mAP50": float(box.ap50[i]),
                "mAP50-95": float(box.ap[i]),
            })

    mp, mr = float(box.mp), float(box.mr)
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    # ── 2. 결과 출력 ─────────────────────────────────────────────────────────
    W = 72
    click.echo("\n" + "=" * W)
    click.echo(f"  평가 결과  |  split={split}  |  {weight_stem}")
    click.echo("=" * W)
    hdr = f"  {'Class':<12} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8} {'mAP50-95':>10}"
    click.echo(hdr)
    click.echo("-" * W)
    for row in rows:
        click.echo(
            f"  {row['class']:<12} {row['P']:>8.4f} {row['R']:>8.4f} {row['F1']:>8.4f}"
            f" {row['mAP50']:>8.4f} {row['mAP50-95']:>10.4f}"
        )
    click.echo("-" * W)
    click.echo(
        f"  {'all (mean)':<12} {mp:>8.4f} {mr:>8.4f} {mf1:>8.4f}"
        f" {box.map50:>8.4f} {box.map:>10.4f}"
    )
    click.echo("=" * W)

    # ── 3. 커스텀 bbox 시각화 저장 ───────────────────────────────────────────
    click.echo(f"\n[Evaluate] 시각화 저장 중 → {vis_dir}")

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        results = model.predict(
            source=str(img_path),
            conf=vis_conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        r0 = results[0]
        specs: list[BBoxSpec] = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy  = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clses = r0.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clses):
                cls_id = int(cls_id)
                color = CLASS_COLORS.get(cls_id, COLOR_ERROR)
                label = f"{names.get(cls_id, str(cls_id))} {c:.2f}"
                specs.append(BBoxSpec(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    lines=[label],
                    color=color,
                ))

        vis = draw_bboxes(img_bgr, specs)
        cv2.imwrite(str(vis_dir / img_path.name), vis)

    click.echo(f"[Evaluate] 시각화 {len(img_paths)}장 저장 완료")
    click.echo(f"[Evaluate] 완료 → {save_dir}")
    return metrics


if __name__ == "__main__":
    cli()
