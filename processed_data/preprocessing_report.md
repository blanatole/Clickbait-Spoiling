# Data Preprocessing Report

**Task Type:** generation
**Output Directory:** processed_data

## Processing Steps

### Spoiler Generation Task
1. **Input Combination**: Combined `postText` and `targetParagraphs` with [SEP] token
2. **Text Cleaning**: Removed HTML, emojis, special characters
3. **Tokenization**: Used GPT-2 Byte-Pair Tokenizer
4. **Output**: Tokenized input-target pairs for sequence-to-sequence training

## Files Generated

- `validation_classification.pkl`: 15,355,510 bytes
- `train_classification.pkl`: 60,938,593 bytes
- `train_generation.jsonl`: 52,195,726 bytes
- `test_generation.jsonl`: 16,224,038 bytes
- `test_classification.pkl`: 19,009,172 bytes
- `preprocessing_report.md`: 0 bytes
- `validation_generation.jsonl`: 13,285,871 bytes

## Next Steps

1. Load processed data for model training
2. Implement baseline models
3. Train and evaluate models
4. Hyperparameter tuning
