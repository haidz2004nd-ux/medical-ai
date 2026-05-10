# Medical QA RAG + Fine-tuning

Du an nay dung domain **y te pho thong**.

## Du lieu

- `raw/`: tai lieu van ban tho.
- `cleaned/`: du lieu da chuan hoa theo tung benh/chu de.
- `qa/train_qa.json`: 525 cap QA dung cho fine-tune.
- `qa/test_qa.json`: 50 cap test thu cong.
- `final_knowledge_base.jsonl`: knowledge base dung cho RAG, gom `page_content` va `metadata`.

## Muc 2: Fine-tuning

File chay: `scripts/02_finetune_qlora_colab.py`

Cau hinh de xuat:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Phuong phap: QLoRA 4-bit
- LoRA rank: 16
- Epoch: 3
- Seed: 42
- Max sequence length: 1024
- Validation split: 10%
- Format train: Qwen chat template
- Train data: `qa/train_qa.json`
- Output adapter: `models/qwen2_5_1_5b_medical_lora`

Cach chay tren Colab Free:

1. Upload thu muc du an len Google Drive, vi du: `/content/drive/MyDrive/medical_qa`.
2. Bat GPU: `Runtime > Change runtime type > T4 GPU`.
3. Mo `scripts/02_finetune_qlora_colab.py`, copy sang notebook hoac upload file `.py` len Colab.
4. Chay cac cell tu tren xuong duoi.
5. Neu Colab het VRAM, doi `BASE_MODEL` thanh `Qwen/Qwen2.5-0.5B-Instruct`.

## Muc 3: Pipeline RAG

File chay: `scripts/03_build_rag_faiss.py`

Chay cai thu vien:

```bash
pip install -r requirements_rag.txt
```

Build FAISS index:

```bash
python scripts/03_build_rag_faiss.py --build
```

Test retrieve top-k va in prompt:

```bash
python scripts/03_build_rag_faiss.py --question "Trieu chung cua viem phoi la gi?"
```

Test retrieve va sinh cau tra loi bang LLM:

```bash
python scripts/03_build_rag_faiss.py --question "Trieu chung cua viem phoi la gi?" --generate
```

Pipeline:

```text
User question
  -> normalize query / detect disease + section
  -> embedding cau hoi
  -> FAISS similarity search + metadata boost
  -> lay top-k=5 chunks
  -> ghep context vao prompt template
  -> LLM sinh cau tra loi
```

Cau hinh:

- Knowledge base: `final_knowledge_base.jsonl`
- Chunking: `RecursiveCharacterTextSplitter`
- `chunk_size=600`
- `chunk_overlap=100`
- Tu dong sua nhanh loi mojibake tieng Viet khi load knowledge base
- Metadata enrichment: dua `disease`, `section`, `topic`, `source` vao text truoc khi embedding
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embedding dimension: 384
- Embedding type: multilingual sentence embedding, optimized for semantic similarity
- Embedding normalization: `normalize_embeddings=True`
- Vector store: FAISS
- Retriever generation: metadata-aware retrieval, `top_k=5`, `fetch_k=30`
- Entity detection: regex whole phrase matching, co normalize tieng Viet khong dau
- Disease synonyms: map cac cach goi nhu `viem phoi`/`pneumonia`, `dai thao duong`/`tieu duong`
- Metadata filter: neu detect duoc disease thi uu tien chunks cua disease do
- Hybrid rerank: FAISS similarity candidates + metadata boost theo `disease` va `section`
- Generation decoding: deterministic, `do_sample=False`
- MMR config co san de tham khao: `k=5`, `fetch_k=30`, `lambda_mult=0.7`
- Retrieval analysis: similarity search with score de phuc vu Recall@5
- Output index: `embeddings/faiss_medical_index_enriched`

Prompt RAG yeu cau mo hinh:

- Chi tra loi dua tren ngu canh retrieve.
- Noi ro khi khong du thong tin trong tai lieu.
- Khong su dung kien thuc ben ngoai ngu canh.
- Khong tu suy dien neu cau tra loi khong co trong ngu canh.
- Khong chan doan thay bac si.
- Khong ke don thuoc.
- Khuyen di kham/cap cuu voi dau hieu nang.

## So sanh 4 cau hinh

File chay: `scripts/04_run_abcd_experiment.py`

Chay experiment 4 cau hinh:

```bash
python scripts/04_run_abcd_experiment.py --max-eval-samples 50
```

Ket qua luu trong `evaluation/`:

- `metrics_summary.csv`
- `metrics_summary.json`
- `human_eval_50_questions.csv`
- `human_eval_50_questions.xlsx`

Bon cau hinh:

| Cau hinh | Mo ta |
|---|---|
| A | LLM goc, khong RAG |
| B | LLM goc + RAG |
| C | LLM fine-tuned, khong RAG |
| D | LLM fine-tuned + RAG |

Script dung `qa/test_qa.json`. Neu chua co, script chi chay smoke test bang vai cau mau.

## Ghi chu nop bai

Du an hien da dat yeu cau train QA `>= 300` vi `qa/train_qa.json` co 525 cap. Phan test da co 50 cap trong `qa/test_qa.json` theo schema:

```json
[
  {
    "instruction": "Cau hoi test",
    "input": "",
    "output": "Dap an chuan"
  }
]
```
