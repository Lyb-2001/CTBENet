⛄ Generative-to-Discriminative: Uncertainty-Aware Distillation for RGB-T Snowy Urban Scene Parsing
This is the official repository for the paper "Generative-to-Discriminative: Uncertainty-Aware Distillation for RGB-T Snowy Urban Scene Parsing".

📢 News
[Coming Soon] The complete source code (training and evaluation scripts) will be fully released upon the acceptance of our paper. Stay tuned!

[202X.XX] Pre-trained weights and visualization results are now available for download.

🔥 Highlights
We pioneer a novel Generative-to-Discriminative (G2D) framework that successfully resolves the trade-off between generative uncertainty modeling and real-time inference in adverse weather conditions.

Our exceptionally lightweight student network, G2DNet-S*, achieves state-of-the-art accuracy while maintaining highly efficient deployment metrics:

🏆 Performance: 83.0% mIoU on the challenging SUS dataset.

⚡ Efficiency: Requires only 5.76 M parameters and 15.62 G FLOPs.

🚀 Speed: Runs at a robust 24.76 FPS, paving the way for real-time autonomous driving applications.

🏗️ Architecture and Details
1. Overall G2D Framework & Uncertainty-Aware Distillation (UAD) Strategy
The proposed framework decouples the complex generative modeling from real-time execution. The UAD strategy bridges the capacity gap between the generative teacher and the discriminative student.

2. Generative Teacher (G2DNet-T)
A powerful generative model that captures robust multimodal semantic distributions and quantifies uncertainty in degraded snowy scenes.

3. Real-Time Student (G2DNet-S) with MSA & FDA
An extremely lightweight discriminative network. It is optimized via distillation to mimic the teacher's robust representations, enhanced by Modality-Specific Adapters (MSA) and Feature Distillation Alignment (FDA).

📊 Results
Quantitative Results (Efficiency vs. Accuracy)
Our G2DNet-S consistently outperforms existing state-of-the-art RGB-T segmentation methods, establishing a new Pareto frontier in the accuracy-efficiency trade-off.

Qualitative Visualization
Robust parsing capabilities in extremely degraded snowy urban environments, effectively suppressing thermal noise and visual ambiguity.

📦 Weights and Visualizations
You can download the pre-trained weights of both the Teacher (G2DNet-T) and Student (G2DNet-S) models, along with full-resolution visualization results, from the link below:

🔗 Baidu Netdisk: Download G2DNet Weights & Results

📂 Base Framework and Dataset
This work is evaluated extensively on the RGB-T Snowy Urban Scene (SUS) dataset. The base framework and the dataset utilized in our research can be obtained from the following repository:

❄️ SUS Dataset: https://github.com/xiaodonguo/SUS_dataset

📖 Citation
If you find our work, code, or pre-trained models useful for your research, please consider citing our paper:

代码段
@article{g2dnet2026,
  title={Generative-to-Discriminative: Uncertainty-Aware Distillation for RGB-T Snowy Urban Scene Parsing},
  author={Li, Yiben and others},
  journal={Coming Soon},
  year={2026}
}
(The bibtex will be updated upon publication)

📧 Contact
If you have any questions, encounter issues, or would like to discuss the paper/code, please feel free to drop me an email at: yibenli2001@163.com.
