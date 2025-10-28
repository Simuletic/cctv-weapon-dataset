Simuletic Weapon Detection Dataset

## Overview
This is an open-source synthetic dataset for computer vision (CV) object detection tasks, focusing on people holding weapons in public areas viewed from CCTV camera perspectives. The dataset consists of high-quality, realistic synthetic images.

Key features:
- **Classes**: "person" and "weapon" (e.g., guns, knives in various poses and scenarios).
- **Annotations**: Bounding boxes (rectangles) in YOLO format for easy integration with models like YOLOv8.
- **Resolution**: Mixed resolutions.
- **Purpose**: Designed for training and evaluating AI models in security, surveillance, and threat detection. Addresses data scarcity and privacy issues with synthetic alternatives.

This dataset is a sample done by Simuletic. We are working on open sourced dataset to help with weapon and threat detection.

For custom scenarios, larger datasets, or videos, visit [simuletic.com](https://simuletic.com).

## Dataset Structure
- **images/**: Folder containing .jpg or .png files (e.g., image001.jpg).
- **labels/**: Folder with YOLO .txt files (one per image, e.g., image001.txt). Each line: `class_id center_x center_y width height` (normalized 0-1).
- **annotations.csv** (optional): A CSV summary with columns like image_name, class, x_min, y_min, x_max, y_max for quick reference.

Example YOLO label line:
0 0.45 0.55 0.20 0.30  # person
1 0.60 0.70 0.10 0.15  # weapon
text## Usage
To load and use the dataset:
1. Clone this repo: `git clone https://github.com/yourusername/simuletic-weapon-detection.git`


Sample dataset.yaml (create in root):
yamlpath: /path/to/dataset
train: images
val: images  # Use same for small datasets
names:
  0: person
  1: weapon

For Hugging Face integration: See our repo on Hugging Face.

Preprocess resolutions if needed (e.g., via OpenCV for resizing).

Ethics and Limitations

This is fully synthetic data—no real individuals or events are depicted.
Intended for ethical use in research/security (e.g., improving detection models). Do not use for harmful purposes.
Potential biases: Scenarios may not cover all real-world diversity; audit for fairness.
For production, combine with real data and validate.

License
This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). You are free to share and adapt it, provided you give appropriate credit to Simuletic.
Citation
If you use this dataset, please cite:
text@dataset{simuletic_weapon_detection_2025,
  author = {Your Name / Simuletic Team},
  title = {Simuletic Synthetic Weapon Detection Dataset},
  year = {2025},
  url = {https://github.com/yourusername/cctv-weapon-dataset}
}
Links

Hugging Face: xxx
Kaggle: xxx
Feedback? Reach out via Simuletic.com or issues here.
