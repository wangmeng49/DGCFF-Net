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

## 📒 Folder Structure

  Prepare the following folders to organize this repo:
  
  DGCFF-Net
          ├── rscd (code)
          ├── work_dirs (save the model weights and training logs)
          │   └─CLCD_BS4_epoch200 (dataset)
          │       └─stnet (model)
          │           └─version_0 (version)
          │              │  └─ckpts
          │              │      ├─test (the best ckpts in test set)
          │              │      └─val (the best ckpts in validation set)
          │              ├─log (tensorboard logs)
          │              ├─train_metrics.txt (train & val results per epoch)
          │              ├─test_metrics_max.txt (the best test results)
          │              └─test_metrics_rest.txt (other test results)
          └── data
              ├── LEVIR_CD
              │   ├── train
              │   │   ├── A
              │   │   │   └── images1.png
              │   │   ├── B
              │   │   │   └── images2.png
              │   │   └── label
              │   │       └── label.png
              │   ├── val (the same with train)
              │   └── test(the same with train)
              ├── WHU_CD
              │   ├── train
              │   │   ├── image1
              │   │   │   └── images1.png
              │   │   ├── image2
              │   │   │   └── images2.png
              │   │   └── label
              │   │       └── label.png
              │   ├── val (the same with train)
              │   └── test(the same with train)
              └── SYSU_CD
                  ├── train
                  │   ├── time1
                  │   │   └── images1.png
                  │   ├── time2
                  │   │   └── images2.png
                  │   └── label
                  │       └── label.png
                  ├── val (the same with train)
                  └── test(the same with train)


---

## ⚠️ Note

Please download the datasets from their official sources.
We do not redistribute the datasets due to copyright restrictions.


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
