# Models Directory

This directory contains trained models and model-related files for the Clickbait Spoiling project.

## Directory Structure

```
models/
├── README.md              # This file
├── gpt2_spoiler/         # GPT-2 model files (to be created)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── training_args.json
├── bert_classifier/      # BERT classification model (planned)
├── t5_generator/         # T5 generation model (planned)
└── ensemble/            # Ensemble model combinations (planned)
```

## Model Types

### 1. GPT-2 Spoiler Generator
- **Purpose**: Generate spoilers for clickbait posts
- **Architecture**: Fine-tuned GPT-2 small (124M parameters)
- **Input**: Clickbait post text + target paragraphs
- **Output**: Generated spoiler text
- **Status**: In development

### 2. BERT Classifier (Planned)
- **Purpose**: Classify spoiler types (phrase/passage/multi)
- **Architecture**: BERT-base-uncased
- **Input**: Post text and spoiler candidates
- **Output**: Spoiler type classification
- **Status**: Planned

### 3. T5 Generator (Planned)
- **Purpose**: Alternative text generation approach
- **Architecture**: T5-small or T5-base
- **Input**: "Generate spoiler: [post text]"
- **Output**: Generated spoiler
- **Status**: Planned

## Model Files

### Saved Model Components
- `config.json`: Model configuration
- `pytorch_model.bin`: Model weights
- `tokenizer.json`: Tokenizer configuration
- `training_args.json`: Training hyperparameters
- `training_log.txt`: Training progress log

### Evaluation Results
- `evaluation_results.json`: Metrics (BLEU, ROUGE, BERTScore)
- `sample_predictions.txt`: Example predictions
- `confusion_matrix.png`: Classification results (if applicable)

## Usage

### Loading a Trained Model
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./models/gpt2_spoiler/')
tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2_spoiler/')

# Generate spoiler
input_text = "Your clickbait post here"
inputs = tokenizer.encode(input_text, return_tensors='pt')
outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
spoiler = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Model Evaluation
```python
from scripts.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model('./models/gpt2_spoiler/', test_data)
print(f"BLEU Score: {results['bleu']}")
print(f"ROUGE Score: {results['rouge']}")
```

## Training Information

### Current Status
- **GPT-2**: Training in progress
- **BERT**: Not started
- **T5**: Not started

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX 3080/4080 or better

### Training Time Estimates
- **GPT-2 Small**: 2-4 hours on GPU, 12-24 hours on CPU
- **BERT Base**: 1-2 hours on GPU, 6-12 hours on CPU
- **T5 Small**: 3-6 hours on GPU, 18-36 hours on CPU

## Notes

- Models are saved in PyTorch format
- All models use Hugging Face Transformers library
- Model files are excluded from git (see .gitignore)
- Download trained models from releases or train locally
- Ensure sufficient disk space (models can be 500MB - 2GB each)
