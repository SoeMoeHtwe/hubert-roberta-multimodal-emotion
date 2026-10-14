# Improving Multimodal Speech Emotion Recognition with Gating Mechanism on HuBERT and RoBERTa Representations

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📌 Abstract
Emotion recognition from speech plays a vital role in human-computer interaction in both industry and academia. To classify emotions from speech, most researchers employ Multimodal Speech Emotion Recognition (MSER) by combining audio and textual features. However, traditional multimodal fusion methods (such as simple late fusion) often fail to dynamically weight the importance of audio and text modalities, leading to suboptimal performance. To address this limitation, this paper proposes an improved Multimodal Speech Emotion Recognition (MSER) framework that leverages HuBERT for robust acoustic feature extraction and RoBERTa for semantic text representation, combined with a Gating Mechanism to dynamically control the contribution of each modality. Moreover, this adaptive gating approach enhances model generalization, ensuring robust performance across diverse acoustic conditions. Evaluating models on challenging real-world datasets like MSPPODCAST is exceptionally difficult due to their in-the-wild complexity, high acoustic variability, and natural conversational ambiguity. Nevertheless, experimental results clearly demonstrate that incorporating a Gating Mechanism significantly outperforms traditional late fusion strategies, achieving an accuracy of 54.36% under these complex real-world conditions.

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/hubert-roberta-multimodal-emotion.git](https://github.com/YOUR_USERNAME/hubert-roberta-multimodal-emotion.git)
cd hubert-roberta-multimodal-emotion

# (Recommended) Create and activate a virtual environment
conda create -n emo_env python=3.10 -y
conda activate emo_env

# Install requirements
pip install -r requirements.txt
```

---

# 🗂️ Dataset
This code has been developed for the MSP-Podcast corpus. Access to the dataset requires a special permission. Therefore, we provide sample ```.csv``` files which show the format compatible with our code.

---

This code has been adapted from [here](https://github.com/msplabresearch/MSP-Podcast_Challenge_IS2025).
