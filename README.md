# 🎯 Clickbait Spoiling Project

## 📋 Tổng quan

Dự án này nghiên cứu về **clickbait spoiling** - việc tạo ra các đoạn văn bản ngắn gọn nhằm thỏa mãn sự tò mò được tạo ra bởi các tiêu đề clickbait. Dự án sử dụng dataset **Webis Clickbait Spoiling Corpus 2022** với 5,000 bài viết clickbait được thu thập từ Facebook, Reddit và Twitter.

## 🎯 Mục tiêu

- 🤖 Xây dựng các mô hình AI để tự động tạo ra spoiler cho các tiêu đề clickbait
- 📊 So sánh hiệu suất của các phương pháp khác nhau: rule-based, classical ML, và deep learning
- 📈 Đánh giá chất lượng spoiler bằng nhiều metrics khác nhau (BLEU, ROUGE, BERTScore)
- 🚀 Triển khai mô hình tốt nhất thành ứng dụng demo

## 📊 Dataset

### Thống kê Dataset
- **Training set**: 3,200 samples  
- **Validation set**: 800 samples
- **Test set**: 1,000 samples

### Loại Spoiler
- **Phrase spoilers**: Câu trả lời ngắn gọn (1,367 samples trong tập train)
- **Passage spoilers**: Đoạn văn bản dài hơn (1,274 samples trong tập train)  
- **Multi spoilers**: Nhiều phần không liên tục (559 samples trong tập train)

### Nền tảng
- **Twitter**: 1,530 samples (train)
- **Reddit**: 1,150 samples (train)
- **Facebook**: 520 samples (train)

## 🛠️ Môi trường phát triển

### Yêu cầu hệ thống
- Python >= 3.7 (đang sử dụng Python 3.10.18)
- Conda environment: `clickbait`
- CUDA support (tùy chọn cho GPU acceleration)

### Thư viện chính
```bash
# Core ML/DL libraries
torch>=1.9.0
transformers>=4.20.0  
sentence-transformers>=2.2.0
scikit-learn>=1.0.0

# Natural Language Processing
nltk>=3.7
spacy>=3.4.0

# Evaluation metrics
sacrebleu>=2.0.0
rouge-score>=0.1.0
bert-score>=0.3.0

# Data processing
pandas>=1.3.0
numpy>=1.21.0
datasets>=2.0.0
```

## 🚀 Cài đặt

1. **Kích hoạt môi trường conda**:
```bash
conda activate clickbait
```

2. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

3. **Tải spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Thiết lập NLTK data**:
```bash
python scripts/setup_nltk.py
```

## 📁 Cấu trúc thư mục

```
clickbait-spoiling/
├── data/                    # Raw dataset
│   ├── README.md
│   ├── train.jsonl         # Training data (3,200 samples)
│   ├── validation.jsonl    # Validation data (800 samples)
│   └── test.jsonl         # Test data (1,000 samples)
├── processed_data/         # Preprocessed data
│   ├── train_generation.jsonl
│   ├── validation_generation.jsonl
│   ├── test_generation.jsonl
│   ├── train_classification.pkl
│   ├── validation_classification.pkl
│   ├── test_classification.pkl
│   └── preprocessing_report.md
├── scripts/                # Utility scripts
│   ├── setup_nltk.py      # Setup NLTK data
│   ├── data_exploration.py # Data exploration and analysis
│   ├── data_preprocessor.py # Data preprocessing
│   ├── evaluator.py       # Model evaluation
│   └── verify_environment.py # Environment verification
├── evaluation_results/     # Model evaluation outputs
├── models/                 # Trained models (to be created)
├── notebooks/             # Jupyter notebooks (to be created)
├── requirements.txt       # Python dependencies
├── preprocess_data.py     # Main preprocessing script
├── train_gpt2_spoiler.py  # GPT-2 training script
├── test_gpt2_model.py     # GPT-2 testing script
└── README.md             # This file
```

## 🔍 Khám phá và xử lý dữ liệu

### 1. Khám phá dữ liệu
```bash
python scripts/data_exploration.py
```

Script này sẽ:
- 📊 Phân tích cấu trúc dataset
- 📈 Tạo visualizations về phân phối dữ liệu
- 💡 Hiển thị ví dụ cho từng loại spoiler
- 💾 Lưu biểu đồ phân tích vào `data_analysis.png`

### 2. Tiền xử lý dữ liệu
```bash
python preprocess_data.py
```

Script này sẽ:
- 🧹 Làm sạch và chuẩn hóa text
- 🔤 Tokenization với GPT-2 tokenizer
- 💾 Lưu dữ liệu đã xử lý vào `processed_data/`
- 📋 Tạo báo cáo preprocessing

## 📈 Phân tích Dataset

### Độ dài Spoiler
- **Trung bình**: 84.6 ký tự (train set)
- **Trung vị**: 43.0 ký tự (train set)
- **Phạm vi**: 2-1,369 ký tự

### Ví dụ theo loại Spoiler

**Phrase Spoilers**:
- Post: "NASA sets date for full recovery of ozone hole"
- Spoiler: "2070"

**Passage Spoilers**:
- Post: "What happens if your new AirPods get lost or stolen, will Apple do anything?"
- Spoiler: "Apple says that if AirPods are lost or stolen, you'll have to buy new ones, just like any other Apple product."

**Multi Spoilers**:
- Post: "Hot Sauce Taste Test: Find out which we named number 1"
- Spoiler: "Sriracha Hot Chili Sauce"

## 🚀 Huấn luyện và đánh giá mô hình

### 1. Huấn luyện mô hình GPT-2
```bash
python train_gpt2_spoiler.py
```

### 2. Kiểm tra mô hình
```bash
python test_gpt2_model.py
```

### 3. Đánh giá chi tiết
```bash
python scripts/evaluator.py
```

## 📊 Kết quả hiện tại

### Tiến độ dự án
- ✅ **Data Exploration**: Hoàn thành phân tích dataset
- ✅ **Data Preprocessing**: Đã xử lý và tokenize dữ liệu
- ✅ **GPT-2 Implementation**: Đã triển khai mô hình GPT-2 cơ bản
- 🔄 **Model Training**: Đang trong quá trình huấn luyện
- ⏳ **Evaluation**: Chờ hoàn thành training
- ⏳ **Deployment**: Kế hoạch tương lai

### Thống kê dữ liệu đã xử lý
- **Training Generation**: 52.2 MB (3,200 samples)
- **Validation Generation**: 13.3 MB (800 samples)
- **Test Generation**: 16.2 MB (1,000 samples)
- **Classification Data**: Đã tokenize và lưu dạng pickle

## 🎯 Kế hoạch tiếp theo

1. **Model Optimization**: Tinh chỉnh hyperparameters
2. **Advanced Models**: Thử nghiệm BERT, T5, và các mô hình khác
3. **Ensemble Methods**: Kết hợp nhiều mô hình
4. **Web Demo**: Tạo giao diện web để demo
5. **Performance Analysis**: Phân tích chi tiết kết quả

## 📚 Tham khảo

- [Webis Clickbait Spoiling Corpus 2022](https://webis.de/data/webis-clickbait-22.html)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**: Giảm batch size trong training script
2. **NLTK data missing**: Chạy `python scripts/setup_nltk.py`
3. **spaCy model missing**: Chạy `python -m spacy download en_core_web_sm`
4. **Import errors**: Kiểm tra environment với `python scripts/verify_environment.py`

### Kiểm tra môi trường
```bash
python scripts/verify_environment.py
```

## 👥 Contributors

- **Sinh viên**: [Tên của bạn]
- **Trường**: Đại học Ngoại ngữ - Tin học TP.HCM (HUFLIT)
- **Khóa luận tốt nghiệp**: Clickbait Spoiling với Deep Learning

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết

## 📞 Liên hệ

- 📧 Email: [your-email@example.com]
- 🐙 GitHub: [your-github-username]
- 📱 LinkedIn: [your-linkedin-profile]
