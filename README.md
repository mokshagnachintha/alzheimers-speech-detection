# Alzheimer's Disease Detection via Speech Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 🚀 Live Demo

**Try our system now**: [**AD Track - Live Speech Analysis Platform**](https://adtrack.onrender.com/)

Upload an audio file of someone describing the Cookie Theft image and get instant Alzheimer's Disease risk assessment!

---

## � Demo Video Walkthrough

**Watch our comprehensive video tutorial** showing the entire system in action:

[![Watch the Demo - Click to Play](https://img.shields.io/badge/▶%20Click%20Here%20to%20Watch%20Demo%20Video-FF0000?style=for-the-badge&logo=youtube)](https://github.com/mokshagnachintha/alzheimers-speech-detection/raw/master/DEMO_WALKTHROUGH.mp4)

**or [Download Video (24.5 MB)](DEMO_WALKTHROUGH.mp4)**

**In this video, you'll see:**
- 🎤 How to upload audio files to the live platform
- 📊 Real-time Alzheimer's risk assessment results
- 📈 Model comparison across different approaches
- 🔍 Feature importance and explainability analysis
- 💡 Practical use cases and interpretation guide

---

## �📋 Overview

This repository contains a **state-of-the-art multimodal deep learning system** for detecting Alzheimer's Disease and Dementia from speech patterns. The project implements two complementary approaches:

1. **Model Comparison Framework**: Systematic evaluation of 16 different machine learning models across multiple modalities
2. **Multimodal AI System**: Advanced tri-branch neural network fusing text, audio, and linguistic features

The system analyzes the **Pitt Corpus (Cookie Theft Task)** dataset, achieving robust dementia detection by combining:
- **Semantic Analysis**: What is said (using DeBERTa v3)
- **Acoustic Features**: How it is said (using Vision Transformer on Mel-Spectrograms)
- **Clinical Biomarkers**: Linguistic patterns (fillers, repetitions, type-token ratio)

---

## 🎯 Key Features & Innovations

### 1. **Comprehensive Model Comparison**
- **Traditional ML** (4 models): SVM, Random Forest, Naive Bayes, Logistic Regression
- **Deep Learning** (2 models): LSTM, BiLSTM
- **Transformers** (7 models): BERT, RoBERTa, XLNet, ALBERT, Clinical BERT, BioBERT, DeBERTa
- **Hybrid Models** (3 models): CNN-LSTM, Ensemble Voting, Stacked Meta-Learning

Each model tested across:
- **Text-Only** modality (TF-IDF features)
- **Audio-Only** modality (acoustic features)
- **Multimodal** fusion (text + audio)

### 2. **Advanced Multimodal Architecture**
- **Cross-Modal Attention**: Multi-head self-attention mechanism fusing three modalities
- **Intermediate Fusion Strategy**: Advanced feature-level fusion vs. simple concatenation
- **Differential Learning Rates**: Pretrained models (2e-5) vs. task layers (1e-3)
- **Data Augmentation**: SpecAugment (audio) + Context-Aware EDA (text)

### 3. **Strict Data Hygiene**
- **Held-out Test Set**: Complete separation between train/validation/test
- **Stratified K-Fold CV**: 5-fold cross-validation on training set
- **Segmentation-Based Audio**: Uses ground-truth participant speech isolation
- **Real-World Validation**: Tests on both perfect transcripts and ASR-generated text

### 4. **Interpretability & Explainability**
- SHAP value analysis for feature importance
- Confusion matrices and ROC-AUC curves for all models
- Precision-Recall curves for imbalanced classification
- Detailed classification reports with per-class metrics

---

## 📊 Model Architecture

### Multimodal System (Tri-Branch Fusion)

```
Text Input          Audio Input         Linguistic Features
    ↓                   ↓                      ↓
DeBERTa             Vision Transformer   MLP + Adapter
    ↓                   ↓                      ↓
BiLSTM            Spectro (64-dim)       (64-dim each)
    ↓                   ↓                      ↓
├─ Multi-Head Self-Attention (learns inter-modal correlations)
    ↓
Fusion (192-dim embedding)
    ↓
Classification → [Control | Dementia]
```

---

## 📊 Results & Performance

### Model Comparison Summary
| Model | Text Accuracy | Audio Accuracy | Multimodal Accuracy |
|-------|---------------|----------------|---------------------|
| SVM | 0.82 | 0.65 | 0.85 |
| Random Forest | 0.79 | 0.68 | 0.83 |
| Naive Bayes | 0.76 | 0.62 | 0.79 |
| Logistic Regression | 0.84 | 0.67 | 0.86 |
| LSTM | 0.86 | 0.70 | 0.88 |
| BiLSTM | 0.87 | 0.72 | 0.89 |
| **DeBERTa Fusion** | **0.91** | **0.78** | **0.93** |
| Ensemble | 0.88 | 0.73 | 0.91 |

### Multimodal System Metrics (5-Fold CV)
- **Accuracy**: 93.2% ± 2.1%
- **Precision**: 94.1% (Dementia), 92.8% (Control)
- **Recall**: 91.5% (Dementia), 94.8% (Control)
- **F1-Score**: 0.927 ± 0.018
- **AUC-ROC**: 0.962 ± 0.013

---

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA 11.8+ (for GPU acceleration)
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/mokshagnachintha/alzheimers-speech-detection.git
cd alzheimers-speech-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Edit the path configuration in notebooks:
```python
INPUT_PATH = r"/path/to/pitt/corpus/THE FINAL"
OUTPUT_PATH = r"./results"
```

---

## 📖 Usage

### 1. **Run Model Comparison Pipeline**
```bash
jupyter notebook model-comparison.ipynb
```
This notebook:
- Loads and preprocesses both modalities
- Trains all 16 models
- Generates comprehensive evaluation metrics
- Saves visualizations (confusion matrices, ROC curves, PR curves)

### 2. **Run Multimodal Detection System**
```bash
jupyter notebook multimodal-dementia-detection.ipynb
```
This notebook:
- Implements tri-branch architecture
- Performs 5-fold cross-validation
- Generates SHAP explainability analysis
- Tests on held-out test set

### 3. **Try Live Web Demo**
Visit: **[AD Track - https://adtrack.onrender.com/](https://adtrack.onrender.com/)**
- Upload audio files
- Get instant predictions
- View confidence scores

---

## 📚 Documentation

### Notebooks

1. **[model-comparison.ipynb](model-comparison.ipynb)**
   - Exhaustive comparison of 16 ML/DL models
   - Data exploration and preprocessing
   - Visualization of model performance across modalities

2. **[multimodal-dementia-detection.ipynb](multimodal-dementia-detection.ipynb)**
   - State-of-the-art tri-branch architecture
   - Advanced feature extraction (linguistic, acoustic, semantic)
   - Explainability analysis (SHAP values)

### Documentation Files

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical deep-dive on all systems
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[QUICK_START.md](QUICK_START.md)** - Commands and setup guide

### Demo Video
[DEMO_WALKTHROUGH.mp4](DEMO_WALKTHROUGH.mp4) - Complete video tutorial showing:
- How to use the notebooks
- Data preparation steps
- Model training process
- Results interpretation

---

## 🔧 Advanced Usage

### Custom Feature Engineering
```python
from feature_extractors import LinguisticExtractor

extractor = LinguisticExtractor()
features = extractor.extract_all({
    'fillers': True,
    'repetitions': True,
    'type_token_ratio': True,
})
```

### Training on Your Own Data
```python
from models import MultimodalFusion
from torch.utils.data import DataLoader

train_loader = DataLoader(your_dataset, batch_size=4, shuffle=True)
model = MultimodalFusion()

optimizer = configure_optimizer(model)
for epoch in range(10):
    train_epoch(model, train_loader, optimizer)
```

### Model Interpretation
```python
import shap

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)
shap.summary_plot(shap_values, test_data)
```

---

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/mokshagnachintha/alzheimers-speech-detection/issues)
- **Questions**: [GitHub Discussions](https://github.com/mokshagnachintha/alzheimers-speech-detection/discussions)
- **Live Demo**: [https://adtrack.onrender.com/](https://adtrack.onrender.com/)

---

## 📖 References

1. Luz et al. (2021). Alzheimer's Disease Dementia Detection using Speech Analysis
2. He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention
3. Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition
4. Park et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
5. MacWhinney (2000). The CHILDES Project: Tools for Analyzing Talk

---

## ⭐ Acknowledgments

- **Pitt Corpus**: DementiaBank, TalkBank, Carnegie Mellon University
- **Frameworks**: PyTorch, Transformers (HuggingFace), scikit-learn, TensorFlow
- **Inspiration**: Clinical speech-language pathology research community

---

<div align="center">

**Made with 💙 for dementia research and clinical diagnostics**

[**Try AD Track Live Demo**](https://adtrack.onrender.com/) | [**View Repository**](https://github.com/mokshagnachintha/alzheimers-speech-detection)

Last Updated: February 2026 | Status: Active Development

</div>
