# Towards multi-modality fusion and prototype-based feature refinement for clinically significant prostate cancer classification in transrectal ultrasound

Implementation of the paper:

**" Towards multi-modality fusion and prototype-based feature refinement for clinically significant prostate cancer classification in transrectal ultrasound "**  
Accepted at *MICCAI 2024*

## 📄 Paper
https://link.springer.com/chapter/10.1007/978-3-031-72086-4_68

## 🧠 Method
### Frame work
- **Feature Extraction:** Two ResNet extract features from different modalities and fuse corrective features using the attention module.
- **Segmentor:** A few shot segmentation paradigm is used to implement the segmentation task to enhance the encoder's ability to extract features. Optimize prototypes with the Prototype Correction Module.
- **Classifier:** Performs classification tasks and generates grad-cam maps to indicate suspicious areas.
  
![image](https://github.com/user-attachments/assets/2fc52ec8-c101-406e-a817-b88b3f7dc69a)
### Module
- **Attention:** Dimensional Attention Module utilizes dimension-wise attention to re-weight the features and fuses the re-weighted features afterwards. Adaptive Spatial Attention Module aggregates the features from two modalities in pixel-level.
- **Prototype Correction:** It refines the support prototype by query features without introducing extra training parameters.

![image](https://github.com/user-attachments/assets/fd0e3e2a-4e50-4dd4-afc2-6c68680de256)
## 📊 Result
![image](https://github.com/user-attachments/assets/9662f9d9-e829-4f96-8e2d-c86792f3a3f6)
## 📊 CAM
![image](https://github.com/user-attachments/assets/b60bf7c3-42de-469f-8ad2-24275cad06e5)

## 🚀 Getting Started

### 1. Clone the repository

```bash
[git clone [https://github.com/2313595986/SmileCode]
cd SmileCode
```

### 2. Pretrain Model
https://drive.google.com/drive/folders/12Ux0R9ljs66JvvTNLAt7q0RxSmbZf6sS?usp=sharing




