# ⛄ Select, Deconfound, and Elicit: A Causal Thermal Benefit Elicitation Network for RGB-T Snowy Urban Scene Parsing

[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-blue.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-SUS-green.svg)](https://github.com/xiaodonguo/SUS_dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the paper **"Select, Deconfound, and Elicit: A Causal Thermal Benefit Elicitation Network for RGB-T Snowy Urban Scene Parsing"**[cite: 2].

---

## 📢 News
* **[Coming Soon]** The complete PyTorch source code, including training, evaluation scripts, and pre-trained models, will be fully released upon the acceptance of our paper. Stay tuned!

---

## 🔥 Highlights

We propose **CTBENet**, a novel framework that reformulates VFM-based cross-modal adaptation from a causal perspective to address the unstable reliability of thermal cues in adverse snowy environments[cite: 2].

Our method systematically tackles thermal reliability through a "Select, Deconfound, and Elicit" paradigm, achieving state-of-the-art robustness and generalization[cite: 2]:
* 🏆 **State-of-the-Art Performance:** Achieves **85.1% mIoU** and **91.8% mF1** on the challenging SUS dataset[cite: 2].
* 🌍 **Robust Generalization:** Consistently establishes new benchmarks across diverse RGB-T scenarios, including **PST900 (89.7%)**, **FMB (68.1%)**, and **MSRS (81.1%)**[cite: 2].
* 🧠 **Causal Adaptation:** Unlocks the power of frozen Vision Foundation Models (DINOv3) without disrupting semantic priors by explicitly regulating thermal noise and confounding effects[cite: 2].

---

## 🏗️ Architecture and Core Components

CTBENet transitions away from indiscriminate encode-then-fuse paradigms toward a principled causal intervention process[cite: 2]. 

![CTBENet Architecture](https://github.com/user-attachments/assets/PLACEHOLDER_FOR_FIGURE_2)
*Figure 1: Overall architecture of CTBENet. Multi-level thermal states are selectively injected into the frozen RGB backbone through MoSAdapter, and the last-stage feature is refined by PCI before decoding[cite: 2].*

### 1. Mixture-of-States Adapter (MoSAdapter) - *Select*
Instead of rigid layer-wise fusion, MoSAdapter treats multi-level thermal states as candidates[cite: 2]. Guided by semantic text prompts, it dynamically routes reliable thermal features into the frozen RGB backbone via an epsilon-greedy strategy, mitigating the interference of thermal noise[cite: 2].

### 2. Prototype Causal Intervention (PCI) - *Deconfound*
To eliminate high-level thermal artifacts (e.g., background heat acting as environmental shortcuts), PCI performs an entropy-guided causal intervention[cite: 2]. It actively subtracts non-semantic confounder contexts specifically in regions exhibiting high spatial uncertainty[cite: 2].

### 3. Counterfactual Thermal Gain (CTG) - *Elicit*
To prevent "modality laziness" where the powerful RGB VFM dominates optimization, CTG explicitly enforces the network to learn the incremental benefits of the thermal modality[cite: 2]. During training, it compares factual predictions with a counterfactual path (T=Ø) to ensure thermal cues actively increase ground-truth confidence[cite: 2].

---

## 📊 Results

### Quantitative Results
CTBENet establishes new state-of-the-art results on the SUS dataset, outperforming recent foundation-model adaptations (like HFIT and TUNI) and traditional fusion networks[cite: 2].

| Method | Backbone | SUS (mIoU) | PST900 (mIoU) | MSRS (mIoU) | FMB (mIoU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| GMNet | ResNet101 | 81.2 | 84.1 | 73.9 | 49.2 |
| CMX | MiT-B2 | 81.2 | 84.9 | 75.3 | 61.1 |
| MiLNet | MiT-B3 | 83.7 | 85.1 | 74.7 | 61.8 |
| TUNI | TUNI-B | 83.9 | 89.1 | 80.7 | 66.3 |
| **CTBENet (Ours)** | **DINOv3 (ConvNeXt-B)** | **85.1** | **89.7** | **81.1** | **68.1** |

### Qualitative Visualization
Our causal paradigm demonstrates superior parsing capabilities under extremely degraded lighting and severe snow conditions[cite: 2].

![Qualitative Results](https://github.com/user-attachments/assets/PLACEHOLDER_FOR_FIGURE_4)

---

## 📂 Base Framework and Dataset

This work is extensively evaluated on the RGB-T Snowy Urban Scene (SUS) dataset[cite: 2]. The dataset and baseline framework can be obtained from:

* ❄️ **SUS Dataset:** [https://github.com/xiaodonguo/SUS_dataset](https://github.com/xiaodonguo/SUS_dataset)

---

## 📖 Citation

If you find our causal adaptation framework or concepts useful for your research, please consider citing our paper:

```bibtex
@inproceedings{ctbenet2026,
  title={Select, Deconfound, and Elicit: A Causal Thermal Benefit Elicitation Network for RGB-T Snowy Urban Scene Parsing},
  author={Anonymous Submission},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
