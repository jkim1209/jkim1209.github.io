# Computer Vision Project: Document Image Classification

**Developed a model to accurately classify various images including document images from diverse industry domains. Achieved 1st place on the leaderboard with Macro F1 score of 0.9692 by combining ViT-SigLIP backbone, class-specific augmentation, Focal Loss, and high-resolution pretrained models.**

## Project Overview

This project develops a model that classifies various images including document images from diverse industry domains into 17 classes.

GitHub Repository: [https://github.com/jkim1209/doc_img_classification](https://github.com/jkim1209/doc_img_classification)

Presentation Slides: [Google Slides](https://docs.google.com/presentation/d/10o12igXX3xXg1zpI-M4KdHMrxI4OToEL/)

## My Role and Contributions

As team leader, I led the entire project from data analysis to modeling and performance optimization.

- Data Analysis & Preprocessing: Identified class imbalance issues and designed and applied optimized data augmentation pipelines according to document/ID card/vehicle types.
- Model Implementation & Experimentation: Experimented with various pretrained models including ViT to explore architectures best suited to the data characteristics.

## Key Technologies and Implementation

### Technologies Used

- Framework: Python, PyTorch, Torchvision, timm, Albumentations, OpenCV
- Models: ViT, ConvNeXt, EfficientNet, etc.

### Core Implementation

- Customized data augmentation pipeline considering class characteristics
- Addressed class imbalance using Focal Loss
- Introduced models pretrained on high-resolution images (512+)
- Implemented 2-stage inference to re-infer samples classified uncertainly in the first stage

<div style="display: flex; gap: 1rem; align-items: flex-start;">
  <img src="/projects/assets/images/05/01.png" alt="Data Augmentation" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="/projects/assets/images/05/02.png" alt="Model Structure" style="flex: 1; max-width: 50%; height: auto;" />
</div>

![Experiment Results](/projects/assets/images/05/03.png)

## Troubleshooting

### Problem: Breaking Performance Limits on High-Resolution Data

**Problem Description**

Despite applying class-specific data augmentation techniques, the Test Macro F1 Score plateaued around 0.85 with no further improvement.

**Solution**

1. Determined that existing models pretrained on low-resolution (224~384px) images couldn't sufficiently capture detailed features important for document images, such as text.
2. Increased input resolution to 512px or higher for the same model but performance actually decreased. Models accustomed to low resolution were more sensitive to noise in high-resolution images.
3. Revised the hypothesis and introduced models pretrained from the start on high-resolution images (512px+). As a result, the Test Macro F1 Score surged to 0.93, and with additional data refinement and Focal Loss application, achieved a final score above 0.95.

## Results and Achievements

### Quantitative Results

Achieved 1st place in the document image classification competition

- Private Macro F1: 0.9692
- Baseline Macro F1: 0.1695

### Learnings and Insights

- Developed the ability to formulate and verify hypotheses while finding root causes of problems in computer vision projects.
- Experienced firsthand how important it is to match pre-training environments with actual task data characteristics for model performance, and that securing high-quality data (Data-centric AI) is as important as model improvement.
- Developed analytical thinking and perseverance by logically tracking root causes of problems and repeatedly setting and verifying hypotheses, rather than just pursuing result improvements.

![Final Results](/projects/assets/images/05/04.png)
