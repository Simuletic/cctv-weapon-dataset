# CCTV Weapon Detection Dataset: free sample (synthetic CCTV, YOLO + COCO)

**Synthetic CCTV images of people carrying handguns, rifles, shotguns and scoped rifles, with YOLO and COCO boxes: 60 free images here, 1,000 in the full dataset.**

![Six synthetic CCTV images with YOLO boxes: a woman holding a handgun in a convenience store, a person carrying a shotgun in a laundromat in monochrome night mode, a woman with a scoped rifle in a fruit and vegetable market, a person with a rifle in a parking garage, a person with a scoped rifle on riverside stairs at night, and a man carrying a shotgun at a gated entrance](assets/hero_grid.jpg)

**Hard negatives**: look-alike objects and no weapon; only `person` is labelled.

![Four hard-negative synthetic CCTV images, cropped around the person, with only person boxes: a woman holding a bag of crisps in an alley, a woman holding a hair dryer on a terrace at night, a man with an open umbrella at a glass entrance, and a man with a folded umbrella in monochrome night footage](assets/hard_negatives.jpg)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Simuletic/cctv-weapon-dataset/blob/main/notebooks/quickstart.ipynb) [![Licence: CC BY-NC 4.0](https://img.shields.io/badge/licence-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE) [![Full dataset: 1,000 images](https://img.shields.io/badge/full%20dataset-1%2C000%20images-0a7cff.svg)](https://simuletic.com/datasets/cctv-weapon-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset)

This free sample has 60 synthetic CCTV images for **weapon detection, gun detection, firearm detection**, with bounding boxes in **YOLO and COCO** format. The images show people carrying handguns, rifles, shotguns and scoped (sniper) rifles, seen from ceiling- and pole-mounted CCTV cameras in shops, warehouses, car parks, lobbies, kitchens and streets, by day and at night. 15 of them are hard negatives: people carrying bottles, umbrellas, bags, phones, tools or nothing at all. Made by [Simuletic](https://simuletic.com/?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset), which builds synthetic CCTV datasets for computer vision.

The sample has the same classes, label rules and folder layout as the full [CCTV Weapon Detection Dataset (1,000 images)](https://simuletic.com/datasets/cctv-weapon-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset), so code you write for the sample runs unchanged on the full set.

## At a glance

| | Free sample (this repo) | [Full dataset](https://simuletic.com/datasets/cctv-weapon-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset) |
|---|---|---|
| Images | 60 (48 train / 12 val) | 1,000 (900 train / 100 val) |
| Hard negatives | 15 | 213 |
| Classes | 5: `person`, `handgun`, `rifle`, `shotgun`, `sniper_rifle` | same |
| Labels | YOLO `.txt` + COCO `.json` | same |
| Resolution | mostly 1536×1024 and 1672×941 JPEG | same |
| Licence | CC BY-NC 4.0 (non-commercial) | research ($149) or commercial ($590), one-time |

## Classes and box counts

| Class | YOLO id | Boxes in sample | Boxes in full 1K |
|---|---|---|---|
| `person` | 0 | 89 | 1,430 |
| `handgun` | 1 | 12 | 201 |
| `rifle` | 2 | 11 | 207 |
| `shotgun` | 3 | 11 | 194 |
| `sniper_rifle` | 4 | 11 | 185 |

**Labelling rules.** Every person in the frame is boxed as `person`, including small, distant and partly hidden people. Each weapon gets one tight box around the visible weapon, with its type as the class. The person holding it also gets a `person` box. Hard-negative images contain only `person` boxes. Boxes are axis-aligned; YOLO coordinates are normalised (0–1), COCO boxes are `[x, y, width, height]` in pixels.

## Folder layout

```
cctv-weapon-dataset/
  data.yaml                        # Ultralytics config: class names and splits
  images/train/  images/val/       # JPEG, metadata stripped
  labels/train/  labels/val/       # YOLO: class x_center y_center width height (0-1), one .txt per image
  annotations/instances_train.json # COCO
  annotations/instances_val.json
  assets/                         # README images
  notebooks/quickstart.ipynb       # Colab: view labels, fine-tune YOLO, run inference
  quickstart.py                    # same steps as a script
```

## Quick start

**In the browser:** open the [Colab notebook](https://colab.research.google.com/github/Simuletic/cctv-weapon-dataset/blob/main/notebooks/quickstart.ipynb). It loads the sample, draws the labels, fine-tunes YOLO11 for a few epochs on a free GPU and runs inference.

**On your machine:**

```bash
git clone https://github.com/Simuletic/cctv-weapon-dataset
cd cctv-weapon-dataset
pip install ultralytics
python quickstart.py --epochs 30
```

`quickstart.py` counts the boxes, saves label previews to `runs/label_previews/`, trains, validates and saves predictions to `runs/predict/`. Point it at the full dataset with `--data /path/to/CCTV_Weapon_Detection_1K/data.yaml`.

**In Python:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=30, imgsz=640)
model.predict("images/val", save=True)
```

Sixty images will not give you a deployable detector. The sample is for checking the format, the camera angles and the label quality against your own footage before you buy or train on the full set.

## Use cases

- gun and rifle detection on CCTV and VMS cameras
- weapon-type classification (handgun vs long gun)
- testing false alarms on look-alike objects
- pre-training before fine-tuning on your own footage

## How the images were made

The images are fully synthetic: there are no real people and no real incident footage in them. Each image was generated to a scene and camera specification (environment, camera height and angle, lighting, distance), labelled automatically, and checked before release: a vision-language model verified that the object is present, that the frame reads as CCTV and that each box sits on the right object, and the set was reviewed by eye. Images that failed were removed.

**Limitations.** Synthetic images do not cover every camera, lens, compression artefact or scene you will meet in deployment. Validate on your own footage, and for production combine synthetic and real data. Some targets are small at CCTV distance; that is deliberate, but it makes the task harder than close-up datasets.

## Get the full dataset

The full **[CCTV Weapon Detection Dataset](https://simuletic.com/datasets/cctv-weapon-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset)** has 1,000 images (213 hard negatives) with the same labels and layout, as a single download. One-time licence: **research $149** (non-commercial, one lab or course) or **commercial $590** (one organisation; models you train are yours). Need other scenes, cameras or classes? See [custom data](https://simuletic.com/custom-data?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset).

Read more: [weapon detection dataset guide on the Simuletic blog](https://simuletic.com/blog/weapon-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset).

## More synthetic CCTV datasets from Simuletic

- [CCTV Knife Detection Dataset](https://github.com/Simuletic/cctv-knife-detection-dataset): free sample on GitHub · [full 1K](https://simuletic.com/datasets/cctv-knife-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset)
- [CCTV Fall Detection Dataset](https://github.com/Simuletic/cctv-fall-detection-dataset): free sample on GitHub · [full 1K](https://simuletic.com/datasets/cctv-fall-incident-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset)
- [All datasets](https://simuletic.com/datasets?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset) · [Free samples on Hugging Face](https://huggingface.co/Simuletic) · [All GitHub repos](https://github.com/Simuletic)

## Licence

The images and labels in this sample are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): you may share and adapt them for non-commercial purposes with credit to Simuletic. The code (`quickstart.py`, `notebooks/`) is MIT-licensed. For commercial use, buy the [commercial licence](https://simuletic.com/datasets/cctv-weapon-detection-dataset?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset). The full dataset is licensed separately.

## Citation

```bibtex
@misc{simuletic_weapon_2026,
  author       = {{Simuletic}},
  title        = {CCTV Weapon Detection Dataset: synthetic CCTV images with YOLO and COCO labels},
  year         = {2026},
  publisher    = {Simuletic},
  howpublished = {\url{https://simuletic.com/datasets/cctv-weapon-detection-dataset}},
  note         = {Free sample: \url{https://github.com/Simuletic/cctv-weapon-dataset}}
}
```

## Contact

Questions, bugs in the labels or custom requests: contact@simuletic.com · [simuletic.com](https://simuletic.com/contact?utm_source=github&utm_medium=readme&utm_campaign=cctv-weapon-dataset) · [open an issue](https://github.com/Simuletic/cctv-weapon-dataset/issues)
