"""
Pipeline RAG cho domain y tế phổ thông (FIXED VERSION)
=======================================================
Các sửa đổi so với bản gốc:
  1. Enriched text: bỏ duplicate metadata trong page_content
  2. retrieve_hybrid: dùng vector score thực thay vì gán 0.0 cho metadata source
  3. Prompt template: ràng buộc chặt hơn, cấm tạo quiz/trắc nghiệm
  4. ask_rag: thêm repetition_penalty, temperature, stopping criteria

Chạy trên Colab:
  1. Upload dự án lên Google Drive.
  2. Sửa PROJECT_DIR nếu cần.
  3. Runtime GPU nếu muốn embedding nhanh hơn.
"""

# %% Cài thư viện
# !pip install -q sentence-transformers faiss-cpu langchain langchain-community \
#              transformers accelerate "bitsandbytes>=0.46.1" peft

# %% Mount Google Drive
# from google.colab import drive
# drive.mount("/content/drive")

import json
import os
import re
import unicodedata

import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ── Cấu hình ────────────────────────────────────────────────────────────────
PROJECT_DIR   = "/content/drive/MyDrive/NPL/newdata/data"
KB_PATH       = os.path.join(PROJECT_DIR, "final_knowledge_base.jsonl")
FAISS_DIR     = os.path.join(PROJECT_DIR, "embeddings", "faiss_medical_index_enriched")

BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RETRIEVER_TOP_K      = 5
RETRIEVER_FETCH_K    = 30
RETRIEVER_LAMBDA_MULT = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — Load knowledge base & chunking
# FIX: Không wrap thêm metadata vào page_content để tránh duplicate.
#      page_content gốc trong JSONL đã có cấu trúc đầy đủ (Bệnh/Mục/Nội dung).
# ══════════════════════════════════════════════════════════════════════════════
raw_docs = []
with open(KB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item     = json.loads(line)
        metadata = item.get("metadata", {})

        # FIX: dùng page_content gốc, không prepend metadata nữa
        raw_docs.append(
            Document(
                page_content=item["page_content"],
                metadata=metadata,
            )
        )

print("Số document gốc:", len(raw_docs))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(raw_docs)

print("Số chunks:", len(chunks))
print(chunks[0].page_content[:500])
print(chunks[0].metadata)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — Embedding & FAISS index
# ══════════════════════════════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

if os.path.exists(FAISS_DIR):
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("Đã load FAISS index có sẵn tại:", FAISS_DIR)
else:
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(FAISS_DIR)
    print("Đã tạo và lưu FAISS index tại:", FAISS_DIR)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": RETRIEVER_TOP_K,
        "fetch_k": RETRIEVER_FETCH_K,
        "lambda_mult": RETRIEVER_LAMBDA_MULT,
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — Tiện ích: normalize, metadata inference, synonyms
# ══════════════════════════════════════════════════════════════════════════════
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("\u0111", "d")


def contains_whole_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase).strip()
    if not normalized_phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


ALL_DISEASES = sorted(
    {doc.metadata.get("disease", "") for doc in chunks if doc.metadata.get("disease", "")},
    key=lambda d: len(normalize_text(d)),
    reverse=True,
)

print("Danh sách disease/entity trong KB:")
for disease in ALL_DISEASES:
    print("-", disease)

DOCS_BY_DISEASE = {
    disease: [doc for doc in chunks if doc.metadata.get("disease") == disease]
    for disease in ALL_DISEASES
}

DISEASE_INDEXES = {
    disease: FAISS.from_documents(docs, embedding_model)
    for disease, docs in DOCS_BY_DISEASE.items()
    if docs
}
print("Đã build local FAISS indexes theo disease:", len(DISEASE_INDEXES))


DISEASE_SYNONYMS = {
    "viem phoi":                              ["pneumonia", "sung phoi", "nhiem trung phoi"],
    "dai thao duong":                          ["tieu duong", "diabetes", "diabetes mellitus"],
    "tang huyet ap":                           ["cao huyet ap", "hypertension"],
    "gout (gut)":                              ["gout", "gut", "thong phong"],
    "benh hen suyen (hen phe quan)":           ["hen suyen", "hen phe quan", "asthma"],
    "hen suyen (hen phe quan)":                ["hen suyen", "hen phe quan", "asthma"],
    "viem xoang":                              ["nhiem trung xoang", "viem mui xoang"],
    "benh viem xoang (viem cac xoang canh mui)": ["viem xoang", "viem mui xoang"],
    "viem hong":                               ["dau hong", "pharyngitis"],
    "viem da day":                             ["dau da day", "gastritis"],
    "suy than":                                ["benh than", "kidney failure"],
    "suy tim":                                 ["heart failure"],
    "tao bon":                                 ["constipation"],
    "tieu chay":                               ["diarrhea"],
    "tram cam":                                ["depression"],
    "roi loan lo au":                          ["lo au", "anxiety"],
    "mat ngu":                                 ["insomnia"],
    "dau dau":                                 ["nhuc dau", "headache"],
    "nhuc dau":                                ["dau dau", "headache"],
    "dau nguc":                                ["tuc nguc", "chest pain"],
    "kho tho":                                 ["shortness of breath", "dyspnea"],
    "ho":                                      ["cough"],
    "ho nhieu":                                ["ho keo dai", "cough"],
}

SECTION_INTENTS = {
    "symptoms":          ["trieu chung", "dau hieu", "bieu hien", "nhan biet"],
    "causes":            ["nguyen nhan", "tai sao", "do dau"],
    "treatment":         ["dieu tri", "chua", "thuoc", "xu tri"],
    "when_to_see_doctor":["khi nao", "di kham", "gap bac si", "cap cuu"],
    "prevention":        ["phong ngua", "phong benh"],
    "definition":        ["la gi", "dinh nghia", "tong quan"],
    "risk_groups":       ["nguy co", "doi tuong"],
    "care_notes":        ["luu y", "cham soc"],
}


def get_disease_aliases(disease: str) -> list[str]:
    normalized = normalize_text(disease)
    aliases = [normalized] + DISEASE_SYNONYMS.get(normalized, [])
    return sorted(set(aliases), key=len, reverse=True)


def infer_query_metadata(question: str):
    normalized_q = normalize_text(question)

    matched_disease = None
    for disease in ALL_DISEASES:
        if any(contains_whole_phrase(normalized_q, alias) for alias in get_disease_aliases(disease)):
            matched_disease = disease
            break

    matched_section = None
    for section, keywords in SECTION_INTENTS.items():
        if any(kw in normalized_q for kw in keywords):
            matched_section = section
            break

    return matched_disease, matched_section


def metadata_boost(doc: Document, matched_disease: str | None, matched_section: str | None) -> float:
    """Trả về giá trị boost dương — sẽ bị trừ khỏi distance để nâng rank."""
    boost = 0.0
    if matched_disease and doc.metadata.get("disease") == matched_disease:
        boost += 2.0
    if matched_section and doc.metadata.get("section") == matched_section:
        boost += 1.0
    topic = doc.metadata.get("topic", "")
    if matched_disease and "ho hap" in normalize_text(topic) and "phoi" in normalize_text(matched_disease):
        boost += 0.3
    return boost


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — Retrieval functions
# FIX: retrieve_hybrid dùng vector score thực thay vì gán 0.0 cho metadata docs.
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_with_scores(question: str, top_k: int = RETRIEVER_TOP_K):
    """Retrieval thuần vector (dùng để phân tích/debug)."""
    matched_disease, _ = infer_query_metadata(question)
    local_vs    = DISEASE_INDEXES.get(matched_disease, vectorstore) if matched_disease else vectorstore
    candidate_n = len(DOCS_BY_DISEASE.get(matched_disease, chunks))  if matched_disease else len(chunks)
    return local_vs.similarity_search_with_score(question, k=min(top_k, candidate_n))


def retrieve_mmr(question: str, top_k: int = RETRIEVER_TOP_K):
    """MMR retrieval trên toàn bộ vectorstore."""
    return vectorstore.max_marginal_relevance_search(
        question,
        k=top_k,
        fetch_k=RETRIEVER_FETCH_K,
        lambda_mult=RETRIEVER_LAMBDA_MULT,
    )


def retrieve_hybrid(question: str, top_k: int = RETRIEVER_TOP_K):
    """
    Kiến trúc chuẩn theo khuyến nghị:
      Full KB vector search → metadata reranking → top-k
    
    Lưu ý score: FAISS L2 → distance, nhỏ hơn = tốt hơn.
    metadata_boost dương → trừ vào distance → nâng rank.
    """
    matched_disease, matched_section = infer_query_metadata(question)

    # ✅ Luôn retrieve trên FULL vectorstore, không routing theo disease
    fetch_n = min(RETRIEVER_FETCH_K, len(chunks))
    vector_candidates = vectorstore.similarity_search_with_score(
        question, k=fetch_n
    )

    # Deduplicate
    seen: set[str] = set()
    deduped = []
    for doc, orig_dist in vector_candidates:  # orig_dist: L2 distance, nhỏ = tốt
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, float(orig_dist)))

    # Metadata-aware reranking: adjusted = distance - boost
    reranked = sorted(
        deduped,
        key=lambda item: item[1] - metadata_boost(
            item[0], matched_disease, matched_section
        ),
    )

    # Build top-k output
    top = []
    for doc, orig_dist in reranked[:top_k]:
        boost    = metadata_boost(doc, matched_disease, matched_section)
        adjusted = orig_dist - boost
        top.append((doc, adjusted, orig_dist))

    docs       = [t[0] for t in top]
    query_meta = {
        "matched_disease": matched_disease,
        "matched_section": matched_section,
        "score_direction": "L2 distance — nhỏ hơn = relevant hơn",
    }
    return docs, top, query_meta


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — Context builder
# ══════════════════════════════════════════════════════════════════════════════
def build_context(retrieved_docs: list[Document]) -> str:
    parts = []
    for idx, doc in enumerate(retrieved_docs, 1):
        disease = doc.metadata.get("disease", "")
        section = doc.metadata.get("section", "")
        source  = doc.metadata.get("source", "")
        url     = doc.metadata.get("url", "")
        parts.append(
            f"[Tài liệu {idx}]\n"
            f"Bệnh: {disease} | Mục: {section}\n"
            f"Nguồn: {source} — {url}\n"
            f"Nội dung: {doc.page_content}"
        )
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 6 — Prompt template
# FIX: thêm lệnh cấm tạo quiz/trắc nghiệm, ép format văn xuôi/bullet.
# ══════════════════════════════════════════════════════════════════════════════
def build_rag_prompt(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý hỏi đáp y tế. Trả lời bằng tiếng Việt.\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. Chỉ dùng thông tin có trong NGỮ CẢNH bên dưới.\n"
                "2. KHÔNG dùng kiến thức bên ngoài NGỮ CẢNH.\n"
                "3. KHÔNG tự suy diễn hoặc bổ sung thông tin không có trong NGỮ CẢNH.\n"
                "4. Nếu không có thông tin phù hợp, trả lời đúng câu:\n"
                "   \"Tôi chưa có đủ thông tin trong tài liệu để trả lời chính xác.\"\n"
                "5. KHÔNG chẩn đoán bệnh.\n"
                "6. KHÔNG kê đơn thuốc.\n"
                "7. Trả lời dạng văn xuôi hoặc danh sách gạch đầu dòng (-).\n"
                "8. TUYỆT ĐỐI KHÔNG tạo câu hỏi trắc nghiệm, quiz, hoặc bài kiểm tra.\n"
                "9. Nếu nhiều tài liệu có thông tin giống nhau, gộp lại thành một câu trả lời.\n"
                "10. Trả lời ngắn gọn, đúng trọng tâm."
            ),
        },
        {
            "role": "user",
            "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7 — Load LLM
# ══════════════════════════════════════════════════════════════════════════════
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 8 — StoppingCriteria tuỳ chỉnh
# FIX: dừng sinh khi gặp các pattern không mong muốn (quiz, trắc nghiệm...).
# ══════════════════════════════════════════════════════════════════════════════
class BadFormatStopper(StoppingCriteria):
    """Dừng generation sớm nếu model bắt đầu tạo quiz/trắc nghiệm."""

    STOP_PHRASES = [
        "trắc nghiệm",
        "câu hỏi trắc nghiệm",
        "chọn đáp án",
        "a) ", "b) ", "c) ", "d) ",   # lựa chọn trắc nghiệm
    ]

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device    = device
        # Encode stop phrases thành token ids
        self._stop_ids = [
            tokenizer.encode(phrase, add_special_tokens=False)
            for phrase in self.STOP_PHRASES
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Chỉ check 30 token cuối để tiết kiệm thời gian
        last_tokens = input_ids[0, -30:].tolist()
        decoded     = self.tokenizer.decode(last_tokens).lower()
        return any(phrase in decoded for phrase in self.STOP_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 9 — Hàm ask_rag chính
# FIX: thêm repetition_penalty, temperature, StoppingCriteria.
# ══════════════════════════════════════════════════════════════════════════════
def ask_rag(
    question:       str,
    llm             = model,
    top_k:          int = RETRIEVER_TOP_K,
    max_new_tokens: int = 300,
):
    # 1. Retrieve
    retrieved_docs, reranked_items, query_meta = retrieve_hybrid(question, top_k=top_k)

    # 2. Build context & prompt
    context = build_context(retrieved_docs)
    prompt  = build_rag_prompt(question, context)

    # 3. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    # 4. Stopping criteria
    stopper = BadFormatStopper(tokenizer, llm.device)

    # 5. Generate
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens      = max_new_tokens,
            do_sample           = False,          # greedy — deterministic
            repetition_penalty  = 1.15,           # FIX: tránh lặp câu
            eos_token_id        = tokenizer.eos_token_id,
            stopping_criteria   = StoppingCriteriaList([stopper]),  # FIX: dừng khi gặp quiz
        )

    # 6. Decode chỉ phần được sinh mới
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer           = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer, retrieved_docs, reranked_items, query_meta


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 10 — Test
# ══════════════════════════════════════════════════════════════════════════════
question = "Triệu chứng của viêm phổi là gì?"
answer, retrieved_docs, reranked_items, query_meta = ask_rag(question)
 
print("=" * 60)
print("CÂU TRẢ LỜI:")
print(answer)
 
print("\n--- Query metadata inferred ---")
print(query_meta)
 
print("\n--- Retrieved docs (hybrid rerank) ---")
for i, (doc, adj_dist, orig_dist) in enumerate(reranked_items, 1):
    boost = orig_dist - adj_dist
    print(f"{i}. orig={orig_dist:.4f} → adjusted={adj_dist:.4f} (boost={boost:.1f})")
    print("  ", doc.metadata)
    print("  ", doc.page_content[:200])
    print()
 
print("\n--- Retrieval analysis (pure vector scores) ---")
for i, (doc, score) in enumerate(retrieve_with_scores(question), 1):
    print(f"{i}. score={score:.4f}", doc.metadata)
    print("  ", doc.page_content[:200])
    print()