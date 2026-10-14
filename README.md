<div align="center">

# 🎙️ Multimodal Speech Emotion Recognition
### HuBERT + RoBERTa with Modality Gating Fusion Mechanism

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview
This repository implements a state-of-the-art **Multimodal Emotion Recognition** system designed for the MSP-Podcast Challenge. It fuses acoustic representations from **HuBERT** and textual semantics from **RoBERTa** using a dynamic **Modality Gating Fusion Mechanism**, optimized with **Focal Loss** to handle severe class imbalances.

---

## 🗂️ Project Structure
```text
hubert-roberta-multimodal-emotion/
├── configs/            # Configuration files
├── src/
│   ├── __init__.py
│   ├── dataset.py      # Dataset loading & preprocessing
│   ├── models.py       # HuBERT + RoBERTa Gated Architecture
│   └── losses.py       # Custom Focal Loss implementation
├── scripts/
│   └── train.py        # Main training & evaluation script
├── requirements.txt    # Project dependencies
└── README.md
