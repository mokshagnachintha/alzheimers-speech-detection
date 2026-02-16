# Alzheimer's Disease Detection via Speech Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)  
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)  
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A state-of-the-art **multimodal deep learning system** for detecting Alzheimer's Disease and Dementia from speech analysis. This project combines advanced NLP, audio processing, and linguistic analysis to achieve **93.2% accuracy** with explainable results.

---

## 🚀 Live Demo & Video

### [Try the Live Platform](https://adtrack.onrender.com/)
Upload an audio file and get instant Alzheimer's risk assessment results with detailed analysis.

### [Watch Demo Video](https://mokshagnachintha.github.io/alzheimers-speech-detection/video-player.html)
Click to watch the complete system walkthrough (24.5 MB, auto-plays)

---

## Key Features

- **Multimodal Analysis**: Combines text transcripts, audio spectrograms, and linguistic features
- **High Accuracy**: 93.2% ± 2.1% with 5-fold cross-validation
- **Explainability**: SHAP analysis and feature importance visualization
- **Clinical Biomarkers**: Detects speech patterns including fillers, repetitions, and linguistic anomalies
- **16 Model Comparison**: Comprehensive evaluation of traditional ML, deep learning, and transformer models
- **Production Ready**: Deployed on [AD Track platform](https://adtrack.onrender.com/)

---

## Architecture Overview

### Tri-Branch Fusion Network

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Audio + Transcript                     │
├─────────────────┬────────────────────────┬──────────────────────┤
│                 │                        │                      │
▼                 ▼                        ▼                      ▼
Text Branch    Audio Branch         Linguistic Branch    Fusion Layer
(DeBERTa)      (ViT Spectrogram)     (Clinical Markers)
    │               │                      │                      │
    ▼               ▼                      ▼                      ▼
BiLSTM          Features               MLP + Adapter        Multi-Head
(256-dim)       (64-dim)               (64-dim)            Attention
    │               │                      │                    │
    └───────────────┴──────────────────────┘                    │
                    │                                             │
                    ▼─────────────────────────────────────────────┘
                    
                Fusion Vector (192-dim)
                    │
                    ▼
            Classification Head
                    │
            ┌───────┴───────┐
            ▼               ▼
        Control        Dementia
```

---

## Performance Results

### Model Comparison Summary

| Model | Text Accuracy | Audio Accuracy | Multimodal |
|-------|:---:|:---:|:---:|
| SVM | 82% | 65% | 85% |
| Random Forest | 79% | 68% | 83% |
| LSTM | 86% | 70% | 88% |
| BiLSTM | 87% | 72% | 89% |
| **DeBERTa Fusion** | **91%** | **78%** | **93%** |
| Ensemble | 88% | 73% | 91% |

### Multimodal System Metrics (Best Model)
- **Accuracy**: 93.2% ± 2.1%
- **Precision**: 94.1% (Dementia), 92.8% (Control)
- **Recall**: 91.5% (Dementia), 94.8% (Control)
- **F1-Score**: 0.927 ± 0.018
- **AUC-ROC**: 0.962 ± 0.013
- **Validation**: 5-fold stratified cross-validation

---

## Project Structure

```
alzheimers-speech-detection/
├── model-comparison.ipynb                 # 16-model evaluation framework
├── multimodal-dementia-detection.ipynb    # Tri-branch architecture implementation
├── DEMO_WALKTHROUGH.mp4                   # Video tutorial (24.5 MB)
├── video-player.html                      # Auto-playing video player
├── requirements.txt                       # Python dependencies
├── ARCHITECTURE.md                        # Technical deep-dive documentation
├── QUICK_START.md                         # Quick setup guide
├── README.md                              # This file
└── .gitignore                             # Git configuration
```

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA 11.8+ (optional, for GPU acceleration)
```

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/mokshagnachintha/alzheimers-speech-detection.git
   cd alzheimers-speech-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run notebooks**
   ```bash
   jupyter notebook model-comparison.ipynb
   jupyter notebook multimodal-dementia-detection.ipynb
   ```

---

## Usage

### Using the Live Platform
1. Visit [AD Track](https://adtrack.onrender.com/)
2. Upload an audio file of someone describing the Cookie Theft image
3. Click "Analyze" to get instant risk assessment
4. View detailed results with confidence scores and explanations

### Using the Notebooks

#### Model Comparison
```python
# Load and preprocess data
# Train 16 different models
# Compare performance across modalities
# Generate evaluation plots and statistics
```

#### Multimodal Detection System
```python
# Text processing with DeBERTa
# Audio processing with Vision Transformer
# Linguistic feature extraction
# Tri-branch fusion and classification
# SHAP explainability analysis
```

---

## What's Inside

### Notebooks

**model-comparison.ipynb**
- Systematic evaluation of 16 models
- Traditional ML, Deep Learning, and Transformer models
- Performance comparison across text, audio, and multimodal inputs
- Confusion matrices, ROC curves, and detailed metrics

**multimodal-dementia-detection.ipynb**
- State-of-the-art tri-branch neural network
- Cross-modal attention mechanism
- Intermediate fusion strategy
- SHAP value analysis and feature importance
- Real-world validation on perfect transcripts and ASR output

### Documentation

**ARCHITECTURE.md** (Complete Technical Guide)
- System architecture and data flow
- Model descriptions and hyperparameters
- Training procedures and optimization
- Data preprocessing and augmentation strategies
- Results analysis and interpretation

**QUICK_START.md**
- Step-by-step setup instructions
- Running individual notebooks
- Understanding outputs

---

## Dataset Information

**Pitt Corpus (Cookie Theft Task)**
- Training: ~150 participants (equally balanced control and dementia)
- Testing: ~50 participants (held-out test set)
- Task: Participants describe the Cookie Theft image from the Boston Diagnostic Aphasia Examination (BDAE)
- Data includes: Audio recordings, manual transcripts, and linguistic annotations

---

## Technologies

- **Deep Learning**: PyTorch 2.0+
- **NLP Models**: DeBERTa v3, BERT, RoBERTa, XLNet, ALBERT, Clinical BERT, BioBERT
- **Vision**: Vision Transformer (ViT) for audio spectrograms
- **Classical ML**: SVM, Random Forest, Logistic Regression, Naive Bayes
- **Audio Processing**: Librosa, torchaudio, Mel-spectrogram features
- **Explainability**: SHAP values for feature importance
- **Visualization**: Matplotlib, Seaborn

---

## Key Findings

1. **Multimodal > Single Modality**: Fusion of text, audio, and linguistic features outperforms individual modalities
2. **DeBERTa Superior for Text**: Pre-trained transformer models significantly outperform traditional NLP features
3. **ViT for Audio**: Vision-based representation learning on spectrograms works effectively for audio classification
4. **Linguistic Features Matter**: Clinical biomarkers contribute important discriminative information
5. **Intermediate Fusion Better**: Combining features before classification outperforms late fusion

---

## Performance Advantages

- High sensitivity (91.5%) for dementia detection - critical for medical applications
- High specificity (94.8%) for control classification - minimizes false positives
- Robust cross-validation (±2.1% std) - reliable estimates
- Explainable predictions (SHAP analysis) - supports clinical decision-making

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{alzheimers_speech_detection,
  title={Alzheimer's Disease Detection via Speech Analysis},
  author={Mokshagna Chintha and Rashmitha P Shetty},
  year={2026},
  url={https://github.com/mokshagnachintha/alzheimers-speech-detection}
}
```

---

## References

- DeBERTa: [He et al., 2021](https://arxiv.org/abs/2006.03654)
- Vision Transformer: [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- Pitt Corpus: [Becker et al., 1994](https://linkinghub.elsevier.com/retrieve/pii/S0093934X84707169)

---

## Support

For questions or issues:
- Open an issue on [GitHub Issues](https://github.com/mokshagnachintha/alzheimers-speech-detection/issues)
- Check existing documentation in ARCHITECTURE.md and QUICK_START.md
- Review the video tutorial at the top of this README

---

## Acknowledgments

- **Pitt Corpus**: For providing the Cookie Theft Task dataset
- **HuggingFace**: For pre-trained transformer models
- **PyTorch Team**: For the excellent deep learning framework
- **Contributors**: Rashmitha P Shetty (@rashmithapshetty)

---

## License

MIT License - See LICENSE file for details

**Note**: This project is for research and educational purposes. Products built using this technology must comply with applicable healthcare regulations and disclaimers.

---

**Last Updated**: February 16, 2026  
**Repository**: https://github.com/mokshagnachintha/alzheimers-speech-detection  
**Live Platform**: https://adtrack.onrender.com/
