# Smarter Self-distillation: Optimizing the Teacher For Surgical Video Applications

This repo serves as reproduction code for the following papers:
* **[Self-Distillation for Surgical Action Recognition](https://arxiv.org/abs/2303.12915)** see, [ArXiV](https://arxiv.org/abs/2303.12915) to be published at MICCAI 2023. To reproduce the results of this paper please check the tag miccai23.
* **[Smarter Self-distillation: Optimizing the Teacher For Surgical Video Applications](https://papers.miccai.org/miccai-2025/paper/1323_paper.pdf)**: This is the recent paper published at the MICCAI 2025 conference, check [preprint](https://papers.miccai.org/miccai-2025/paper/1323_paper.pdf)

The new version with reproduction code for the [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/1323_paper.pdf) addresses a reproducibility issue found in the [MICCAI 2023 paper](https://arxiv.org/abs/2303.12915) which used self-distillation with a sub-optimal teacher (without teacher selection). This updated release introduces two key features:
* A teacher selection step to identify the best teacher model (checkpoint) for distillation
* A multi-teacher ensemble approach to improve the quality of soft labels

![](./figures/main_miccai25.png)

<!-- <p align="center">
  <img src="./figures/concept_overview.png" alt="Figure">
</p> -->


## THE CODE WILL BE PUBSLIHED SHORTLY AFTER THE MICCAI 2025 CONFERENCE

## Reproduce the MICCAI 2023 paper **[Self-Distillation for Surgical Action Recognition](https://arxiv.org/abs/2303.12915)**:
* The code is available in the tag: miccai23
* For documentation on the miccai 2023 paper, see [readme_miccai23.md](readme_miccai23.md).
