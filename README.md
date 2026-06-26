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
<img width="611" height="392" alt="image" src="https://github.com/user-attachments/assets/9aa2b098-6053-450d-9960-95a22c50a4c8" />

### 1. Mixture-of-States Adapter (MoSAdapter) - *Select*
Instead of rigid layer-wise fusion, MoSAdapter treats multi-level thermal states as candidates[cite: 2]. Guided by semantic text prompts, it dynamically routes reliable thermal features into the frozen RGB backbone via an epsilon-greedy strategy, mitigating the interference of thermal noise[cite: 2].

### 2. Prototype Causal Intervention (PCI) - *Deconfound*
To eliminate high-level thermal artifacts (e.g., background heat acting as environmental shortcuts), PCI performs an entropy-guided causal intervention[cite: 2]. It actively subtracts non-semantic confounder contexts specifically in regions exhibiting high spatial uncertainty[cite: 2].

<img width="299" height="194" alt="image" src="https://github.com/user-attachments/assets/6b9a8cbc-0842-47a3-813b-2f9e0910534d" />

### 3. Counterfactual Thermal Gain (CTG) - *Elicit*
To prevent "modality laziness" where the powerful RGB VFM dominates optimization, CTG explicitly enforces the network to learn the incremental benefits of the thermal modality[cite: 2]. During training, it compares factual predictions with a counterfactual path (T=Ø) to ensure thermal cues actively increase ground-truth confidence[cite: 2].

---

## 📊 Results

### Quantitative Results
CTBENet establishes new state-of-the-art results on the SUS dataset, outperforming recent foundation-model adaptations (like HFIT and TUNI) and traditional fusion networks[cite: 2].
<img width="597" height="251" alt="image" src="https://github.com/user-attachments/assets/3eb0cd94-8123-450d-980b-bd708135b29e" />
<img width="293" height="310" alt="image" src="https://github.com/user-attachments/assets/7cc40a25-5088-4d40-8c0e-2ec6cfc73b20" />
<img width="293" height="281" alt="image" src="https://github.com/user-attachments/assets/2769d61b-217c-490d-b733-1fb8ecd31977" />


### Qualitative Visualization
Our causal paradigm demonstrates superior parsing capabilities under extremely degraded lighting and severe snow conditions[cite: 2].

<img width="541" height="295" alt="image" src="https://github.com/user-attachments/assets/48515fc6-3887-4e60-b96d-da352661d708" />


---

## 📂 Base Framework and Dataset

This work is extensively evaluated on the RGB-T Snowy Urban Scene (SUS) dataset[cite: 2]. The dataset and baseline framework can be obtained from:

* ❄️ **SUS Dataset:** [https://github.com/xiaodonguo/SUS_dataset](https://github.com/xiaodonguo/SUS_dataset)

---
## Code Release

The complete source code will be released upon the acceptance of our paper. Stay tuned!
