# Hệ thống Hỏi đáp Tiếng Việt trên Miền Tri thức Y Tế với RAG + Fine-tuning LLM

Dự án xây dựng hệ thống hỏi đáp thông minh cho domain y tế phổ thông bằng tiếng Việt, kết hợp Retrieval-Augmented Generation (RAG) và fine-tuning Large Language Model (LLM).

## 📋 Tổng quan

Hệ thống sử dụng:
- **LLM**: Qwen/Qwen2.5-1.5B-Instruct với LoRA fine-tuning
- **RAG Pipeline**: Chunking + Embedding (sentence-transformers) + FAISS vector store
- **Experiments**: So sánh 4 cấu hình (Base vs Fine-tuned, với/không RAG)
- **Đánh giá**: BLEU, ROUGE-L, BERTScore, Recall@5, Human evaluation
- **Demo**: Web UI với React frontend và FastAPI backend

## 🏗️ Kiến trúc

```
├── data/                          # Dữ liệu và scripts xử lý
│   ├── scripts/                   # Scripts Python cho RAG và server
│   ├── qa/                        # Cặp QA cho train/test
│   ├── evaluation/                # Kết quả đánh giá
│   ├── models/                    # Model đã fine-tune
│   ├── embeddings/                # FAISS index
│   └── cleaned/                   # Dữ liệu knowledge base đã xử lý
├── medical-ai-ui/                 # Frontend React
│   ├── src/                       # Source code React
│   ├── public/                    # Static assets
│   └── package.json               # Dependencies Node.js
└── requirements.txt               # Dependencies Python
```

## 🚀 Cài đặt và Chạy

### 1. Clone repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Backend (Python)
```bash
# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# pip install -r requirements.txt

# Chạy server
python data/scripts/server.py
```

### 3. Frontend (React)
```bash
cd medical-ai-ui
npm install
npm start
```

### 4. Truy cập
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

## 📊 Experiments và Đánh giá

### 4 Cấu hình so sánh:
- **A**: Base LLM, không RAG
- **B**: Base LLM, có RAG
- **C**: Fine-tuned LLM, không RAG
- **D**: Fine-tuned LLM, có RAG

### Metrics:
- BLEU-4, ROUGE-L, BERTScore F1
- Recall@5 (cho retrieval)
- Human evaluation (50 câu hỏi)

Chạy experiments: `jupyter notebook data/Danh_gia_model.ipynb`

## 🔧 API Endpoints

- `POST /ask`: Hỏi đáp với RAG
- `GET /metrics_summary`: Lấy kết quả đánh giá

## 📈 Kết quả

| Config | RAG | BLEU-4 | ROUGE-L | BERTScore F1 | Recall@5 |
|--------|-----|--------|---------|--------------|----------|
| A (Base, No RAG) | Không | 0.0248 | 0.2717 | 0.8353 | N/A |
| B (Base, RAG) | Có | 0.1373 | 0.3993 | 0.8634 | 0.84 |
| C (FT, No RAG) | Không | 0.0475 | 0.2982 | 0.8469 | N/A |
| D (FT, RAG) | Có | 0.2161 | 0.4167 | 0.8689 | 0.84 |

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License

## 👥 Tác giả

- [Đỗ Thanh Tú] - [52200240@student.tdtu.edu.vn]
- [Lê Thành Khang] - [52200161@student.tdtu.edu.vn]
- [Nguyễn Châu Chí Hải] - [52200176@student.tdtu.edu.vn]

## 🙏 Lời cảm ơn

- Qwen team cho model Qwen2.5
- Sentence Transformers cho embedding model
- LangChain và FAISS cho RAG pipeline