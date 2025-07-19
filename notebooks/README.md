# Notebooks Directory

This directory contains Jupyter notebooks for interactive analysis, experimentation, and visualization of the Clickbait Spoiling project.

## Directory Structure

```
notebooks/
├── README.md                    # This file
├── 01_data_exploration.ipynb   # Dataset exploration and analysis
├── 02_preprocessing.ipynb      # Data preprocessing experiments
├── 03_model_training.ipynb     # Model training experiments
├── 04_evaluation.ipynb         # Model evaluation and comparison
├── 05_visualization.ipynb      # Results visualization
└── experiments/                # Experimental notebooks
    ├── gpt2_experiments.ipynb
    ├── bert_experiments.ipynb
    └── ensemble_experiments.ipynb
```

## Notebook Descriptions

### 1. Data Exploration (`01_data_exploration.ipynb`)
- **Purpose**: Interactive dataset analysis
- **Contents**:
  - Dataset statistics and distributions
  - Spoiler type analysis
  - Platform-wise data breakdown
  - Text length analysis
  - Sample visualization
- **Status**: To be created

### 2. Preprocessing (`02_preprocessing.ipynb`)
- **Purpose**: Data preprocessing experiments
- **Contents**:
  - Text cleaning techniques
  - Tokenization experiments
  - Data augmentation strategies
  - Quality assessment
- **Status**: To be created

### 3. Model Training (`03_model_training.ipynb`)
- **Purpose**: Interactive model training
- **Contents**:
  - Hyperparameter tuning
  - Training progress visualization
  - Loss curve analysis
  - Model comparison
- **Status**: To be created

### 4. Evaluation (`04_evaluation.ipynb`)
- **Purpose**: Comprehensive model evaluation
- **Contents**:
  - Metric calculations (BLEU, ROUGE, BERTScore)
  - Qualitative analysis
  - Error analysis
  - Performance comparison
- **Status**: To be created

### 5. Visualization (`05_visualization.ipynb`)
- **Purpose**: Results visualization and reporting
- **Contents**:
  - Performance charts
  - Sample predictions
  - Attention visualizations
  - Final report generation
- **Status**: To be created

## Experimental Notebooks

### GPT-2 Experiments
- Fine-tuning experiments
- Architecture modifications
- Generation parameter tuning

### BERT Experiments
- Classification experiments
- Feature extraction
- Transfer learning approaches

### Ensemble Experiments
- Model combination strategies
- Voting mechanisms
- Performance optimization

## Getting Started

### Prerequisites
```bash
# Install Jupyter
pip install jupyter ipywidgets

# Install additional visualization libraries
pip install plotly seaborn matplotlib wordcloud
```

### Running Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

### Recommended Extensions
- **Variable Inspector**: Monitor variables
- **Table of Contents**: Navigate large notebooks
- **Code Folding**: Organize code sections
- **Plotly**: Interactive visualizations

## Best Practices

### Notebook Organization
1. **Clear titles and descriptions** for each section
2. **Markdown documentation** explaining methodology
3. **Code comments** for complex operations
4. **Output clearing** before committing to git
5. **Modular functions** for reusable code

### Data Handling
- Load data from `../data/` and `../processed_data/`
- Save intermediate results to avoid recomputation
- Use relative paths for portability
- Document data transformations

### Visualization Guidelines
- Use consistent color schemes
- Include proper labels and titles
- Save important plots as PNG/SVG files
- Make plots publication-ready

## Integration with Scripts

### Using Project Scripts in Notebooks
```python
import sys
sys.path.append('../scripts/')

from data_preprocessor import DataPreprocessor
from evaluator import ModelEvaluator

# Use project functions
preprocessor = DataPreprocessor()
data = preprocessor.load_data('../data/train.jsonl')
```

### Exporting Notebook Results
```python
# Save results for use in main scripts
import pickle
import json

# Save processed data
with open('../processed_data/notebook_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save configuration
with open('../config/notebook_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Notes

- Notebooks are for experimentation and analysis
- Production code should be in Python scripts
- Clear outputs before committing to git
- Use version control for important experiments
- Document findings and insights in markdown cells
