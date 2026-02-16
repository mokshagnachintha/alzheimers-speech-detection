# Contributing to Alzheimer's Speech Detection Project

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this research project.

## Code of Conduct

- Be respectful and professional in all interactions
- Focus on the scientific accuracy and ethical implications
- Respect patient privacy and data sensitivity
- Acknowledge contributions from collaborators

## Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/yourusername/alzheimers-speech-detection.git
cd alzheimers-speech-detection
```

### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 3. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development Workflow

### Writing Code
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write meaningful commit messages

### Example Docstring
```python
def extract_linguistic_features(text: str) -> Dict[str, float]:
    """
    Extract clinical linguistic biomarkers from transcript.
    
    Args:
        text: CHAT-formatted transcript string
        
    Returns:
        Dictionary with keys: fillers, repetitions, maze_count,
        type_token_ratio, speech_rate, pause_frequency
        
    Example:
        >>> features = extract_linguistic_features("the [/] the mother")
        >>> features['repetitions']
        1
    """
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_feature_extraction.py

# With coverage
pytest --cov=src tests/
```

## Contribution Areas

### High Priority
- [ ] **Real-time Inference API**: REST API for deployment
- [ ] **Additional Audio Models**: Wav2Vec 2.0, HuBERT integration
- [ ] **Performance Optimization**: Model quantization, edge deployment
- [ ] **Explainability**: LIME, SHAP integration
- [ ] **Multilingual Support**: Extend beyond English

### Medium Priority
- [ ] **Web Interface**: User-friendly diagnostic tool
- [ ] **Mobile Application**: iOS/Android companion
- [ ] **Dataset Expansion**: Support for additional corpora
- [ ] **Benchmark Suite**: Standardized evaluation metrics
- [ ] **Documentation**: Improve tutorials and examples

### Low Priority
- [ ] **Visualization Dashboard**: TensorBoard integration
- [ ] **Research Extensions**: Novel architectures, training strategies
- [ ] **Code Examples**: Jupyter notebooks showcasing features
- [ ] **Performance Comparisons**: Comparison with other systems

## Submission Guidelines

### Before Submitting
1. **Test thoroughly**: Run all tests locally
2. **Update documentation**: Reflect changes in README/docs
3. **Check code style**: Run linter (`flake8`, `black`)
4. **Verify reproducibility**: Include random seeds

### Creating Pull Request
1. **Descriptive title**: "Add Wav2Vec 2.0 audio encoder"
2. **Detailed description**: 
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
3. **Link related issues**: "Closes #42"
4. **Include benchmark results**: Performance metrics before/after

### Review Process
- At least one maintainer review required
- CI/CD tests must pass
- Suggestions for improvement treated collaboratively
- Final approval and merge

## Common Contributions

### Adding a New Model

**File: `models/new_model.py`**
```python
import torch
import torch.nn as nn

class NewModelBranch(nn.Module):
    """
    Novel architecture for [modality] processing.
    
    Citation: Author et al. (2024). "Paper Title"
    """
    
    def __init__(self, input_dim: int, output_dim: int = 64):
        """Initialize the model branch."""
        super().__init__()
        self.encoder = nn.Sequential(...)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.encoder(x)
```

**Update: `models/__init__.py`**
```python
from .new_model import NewModelBranch

__all__ = ['NewModelBranch']
```

**Test: `tests/test_new_model.py`**
```python
import torch
from models import NewModelBranch

def test_new_model_forward_pass():
    model = NewModelBranch(input_dim=256)
    x = torch.randn(4, 256)  # batch_size=4
    output = model(x)
    
    assert output.shape == (4, 64)
    assert not torch.isnan(output).any()
```

### Extending Evaluation Metrics

**File: `utils/evaluation.py`**
```python
def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute sensitivity and specificity.
    
    Sensitivity = TP / (TP + FN) = Recall
    Specificity = TN / (TN + FP)
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {'sensitivity': sensitivity, 'specificity': specificity}
```

### Improving Data Pipeline

**Areas:**
- Faster preprocessing
- Better data augmentation
- Additional feature extraction
- Dataset validation

**Example:**
```python
class ImprovedSpectrogram:
    """Enhanced spectrogram generation with zero-crossing rate."""
    
    def __call__(self, audio_path: str) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Existing: Mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # NEW: Add zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Combine features
        return np.vstack([S, zcr])
```

## Code Review Checklist

### For Reviewers
- [ ] Code follows PEP 8 style
- [ ] Tests pass and cover new functionality
- [ ] Documentation is clear and complete
- [ ] No hardcoded paths or credentials
- [ ] Reproducibility ensured (random seeds)
- [ ] Backward compatibility maintained
- [ ] Performance impact assessed

### For Authors
- [ ] All tests pass locally
- [ ] Docstrings complete
- [ ] No debug print statements
- [ ] Logging used appropriately
- [ ] Type hints present
- [ ] README updated if needed
- [ ] Acknowledged related work/citations

## Testing Requirements

### Unit Tests
Test individual functions in isolation
```python
def test_extract_fillers():
    text = "uh the mother uh is washing"
    fillers = extract_fillers(text)
    assert fillers == 2
```

### Integration Tests
Test component interactions
```python
def test_multimodal_forward_pass():
    model = MultimodalFusion()
    batch = create_dummy_batch()
    outputs = model(batch['input_ids'], batch['pixel_values'], ...)
    assert outputs.shape == (4, 2)  # batch_size=4, num_classes=2
```

### Regression Tests
Ensure performance doesn't degrade
```python
def test_model_accuracy_threshold():
    accuracy = evaluate_model(model, test_set)
    assert accuracy > 0.92  # Minimum acceptable threshold
```

## Documentation Standards

### Docstring Format
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    One-line summary of what the function does.
    
    More detailed description explaining the algorithm,
    assumptions, and any important notes.
    
    Args:
        param1: Description of param1 and its type
        param2: Description of param2 and its type
        
    Returns:
        Description of return value and its type
        
    Raises:
        ValueError: When this condition occurs
        RuntimeError: When that condition occurs
        
    Example:
        >>> result = function_name(val1, val2)
        >>> print(result)
        expected_output
        
    Notes:
        Any additional important information or limitations
        
    References:
        Author et al. (2024). "Paper Title". Journal.
    """
```

## Release Process

1. **Version Bump**:
   - Patch: `1.0.0` → `1.0.1` (bug fixes)
   - Minor: `1.0.0` → `1.1.0` (new features)
   - Major: `1.0.0` → `2.0.0` (breaking changes)

2. **Changelog Entry**:
   ```markdown
   ## v1.1.0 (2026-03-01)
   
   ### Features
   - Add Wav2Vec 2.0 audio encoder (#42)
   - Support multilingual transcripts (#38)
   
   ### Bug Fixes
   - Fix memory leak in SpecAugment (#51)
   - Correct TTR calculation for short transcripts (#49)
   
   ### Documentation
   - Add architecture diagram (#47)
   - Improve setup instructions (#46)
   ```

3. **GitHub Release**: Tag release with `git tag v1.1.0`

## Questions or Need Help?

- **Issues**: Open a GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **GitHub**: [@rashmithapshetty](https://github.com/rashmithapshetty)
- **Email**: Contact maintainers at project email

## Team

This project is maintained by:
- **Rashmitha P Shetty** - [@rashmithapshetty](https://github.com/rashmithapshetty)

## Recognition

All contributors are acknowledged in:
- README.md Contributors section
- Releases notes
- Annual recognition post

Thank you for making this research better! 🙏

