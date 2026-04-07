# DGCFF-Net

This repository contains the official implementation of the paper:

**"Difference-Guided and Contextual Feature Fusion for Efficient Change Detection in Remote Sensing Imagery"**

submitted to *The Visual Computer*.

---

## 📌 Overview

DGCFF-Net is a lightweight deep learning network for remote sensing change detection.

It introduces:

* Difference-Guided Fusion (DGF)
* Contextual Feature Fusion (CFF)

## 🧠 Network Architecture
![Network Architecture](DGCFFNet.png)

The model achieves strong performance with only **14.65M parameters** and **11.88G FLOPs**.

---

## ⚙️ Environment

* Python: 3.9
* PyTorch: 2.0.0
* CUDA: 11.7

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

We evaluate our method on three publicly available remote sensing change detection datasets:

### 1. LEVIR-CD

A large-scale dataset for building change detection.
It contains 637 image pairs with a resolution of 1024×1024.

Download link: https://justchenhao.github.io/LEVIR/

---

### 2. SYSU-CD

A dataset focusing on urban change detection with diverse scenarios.

Download link: https://github.com/liumency/SYSU-CD

---

### 3. WHU-CD

A high-resolution dataset for building change detection.

Download link: http://gpcv.whu.edu.cn/data/building_dataset.html

---

## 📌 Data Preprocessing

* All images are cropped into **256×256 patches**
* Data augmentation includes:

  * Random flipping
  * Rotation
* Normalization follows standard settings in remote sensing change detection

---

## 📁 Directory Structure

Please organize the project directory as follows:

```
DGCFF-Net/
├── rscd/                         # source code
├── work_dirs/                    # training outputs
│   └── CLCD_BS4_epoch200/
│       └── stnet/
│           └── version_0/
│               ├── ckpts/
│               │   ├── test/     # best checkpoints on test set
│               │   └── val/      # best checkpoints on validation set
│               ├── log/          # TensorBoard logs
│               ├── train_metrics.txt
│               ├── test_metrics_max.txt
│               └── test_metrics_rest.txt
└── data/
    ├── LEVIR_CD/
    │   ├── train/
    │   │   ├── A/
    │   │   ├── B/
    │   │   └── label/
    │   ├── val/
    │   └── test/
    ├── WHU_CD/
    │   ├── train/
    │   │   ├── image1/
    │   │   ├── image2/
    │   │   └── label/
    │   ├── val/
    │   └── test/
    └── SYSU_CD/
        ├── train/
        │   ├── time1/
        │   ├── time2/
        │   └── label/
        ├── val/
        └── test/
```

---

## 📌 Notes

* `rscd/` contains the implementation of DGCFF-Net
* `work_dirs/` stores model checkpoints, logs, and evaluation results
* `data/` contains all datasets organized in a unified structure

Make sure the dataset paths match the configuration files before training.


---

## 🚀 Training

Example:

```bash
python train.py -c configs/DGCFFNet.py
```

---

## 🔍 Testing

```bash
python test.py -c configs/STNet.py \
  --ckpt work_dirs/CLCD_BS4_epoch200/stnet/version_1/ckpts/test/test_change_f1 \
  --output_dir work_dirs/CLCD_BS4_epoch200/stnet/version_1/ckpts/test \
```

---

## 📊 Reproducibility

The results reported in our paper can be reproduced using this codebase with the provided settings.

---

## 📎 Code

GitHub: https://github.com/你的仓库链接

---

## 📖 Citation

If you find this work useful, please cite our paper.

---

## ⚠️ Note

This code is directly related to the manuscript submitted to *The Visual Computer*.
