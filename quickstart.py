"""Quick start for a Simuletic CCTV dataset: count labels, preview them, fine-tune YOLO, run inference.

    pip install ultralytics
    python quickstart.py                         # this sample, 30 epochs
    python quickstart.py --epochs 5 --imgsz 640  # faster
    python quickstart.py --data /path/to/full_dataset/data.yaml --model yolo11s.pt --imgsz 960

Works for the free sample and for the full 1K dataset (same layout). MIT licence.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

COLORS = [(60, 160, 255), (255, 70, 70), (255, 140, 0), (255, 60, 200), (190, 90, 255), (40, 200, 120)]


def load_config(data: Path) -> tuple[Path, dict]:
    """Return a data.yaml with an absolute dataset path, so training works from any directory."""
    cfg = yaml.safe_load(data.read_text())
    root = data.resolve().parent
    if cfg.get("path") not in (None, "", "."):
        root = (root / cfg["path"]).resolve()
    cfg["path"] = str(root)
    fixed = root / "runs" / "data_abs.yaml"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    fixed.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return fixed, cfg


def label_file(img: Path) -> Path:
    parts = list(img.parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(img: Path) -> list[tuple[int, float, float, float, float]]:
    p = label_file(img)
    if not p.exists():
        return []
    return [(int(a[0]), *map(float, a[1:5])) for a in (l.split() for l in p.read_text().splitlines()) if a]


def summarise(cfg: dict) -> None:
    names = cfg["names"]
    names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    cfg["names"] = names
    for split in ("train", "val"):
        imgs = sorted((Path(cfg["path"]) / cfg[split]).glob("*.jpg"))
        boxes = Counter(names[c] for i in imgs for c, *_ in read_labels(i))
        neg = sum(1 for i in imgs if not {names[c] for c, *_ in read_labels(i)} - {"person"})
        print(f"{split:5s}: {len(imgs):4d} images, {neg} with no target object, boxes {dict(boxes)}")


def preview(cfg: dict, n: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((Path(cfg["path"]) / cfg["val"]).glob("*.jpg"))
    for img in random.Random(0).sample(imgs, min(n, len(imgs))):
        im = Image.open(img).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for c, x, y, bw, bh in read_labels(img):
            box = [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
            d.rectangle(box, outline=COLORS[c % len(COLORS)], width=3)
            d.text((box[0] + 3, box[1] - 12), cfg["names"][c], fill=COLORS[c % len(COLORS)])
        im.save(out / img.name)
    print(f"label previews: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="any Ultralytics detection checkpoint")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu or mps; default picks automatically")
    ap.add_argument("--previews", type=int, default=8, help="label previews to draw (0 = none)")
    ap.add_argument("--skip-train", action="store_true", help="only count and preview labels")
    a = ap.parse_args()

    data, cfg = load_config(a.data)
    runs = Path(cfg["path"]) / "runs"
    summarise(cfg)
    if a.previews:
        preview(cfg, a.previews, runs / "label_previews")
    if a.skip_train:
        return

    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=str(runs), name="train", exist_ok=True, plots=True)
    metrics = model.val(data=str(data), imgsz=a.imgsz, batch=a.batch, device=a.device,
                        project=str(runs), name="val", exist_ok=True)
    print(f"val mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")
    for name, ap50 in zip(cfg["names"], metrics.box.ap50):
        print(f"  {name:15s} AP50 {ap50:.3f}")
    model.predict(source=str(Path(cfg["path"]) / cfg["val"]), imgsz=a.imgsz, conf=0.25, device=a.device,
                  save=True, project=str(runs), name="predict", exist_ok=True)
    print(f"predictions: {runs / 'predict'}")


if __name__ == "__main__":
    main()
