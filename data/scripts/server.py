"""
Medical AI RAG API Server
==========================
Sử dụng FastAPI để tạo API endpoint cho frontend.
"""

import json
import os
import re
import unicodedata
from typing import List, Dict, Any

import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from pydantic import BaseModel

# ── Cấu hình ────────────────────────────────────────────────────────────────
PROJECT_DIR   = os.path.dirname(os.path.dirname(__file__))  # data/
KB_PATH       = os.path.join(PROJECT_DIR, "final_knowledge_base.jsonl")
FAISS_DIR     = os.path.join(PROJECT_DIR, "embeddings", "faiss_medical_index_enriched")

BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RETRIEVER_TOP_K      = 5
RETRIEVER_FETCH_K    = 30
RETRIEVER_LAMBDA_MULT = 0.7

# Global variables
vectorstore = None
retriever = None
model = None
tokenizer = None
chunks = []
ALL_DISEASES = []
DOCS_BY_DISEASE = {}
DISEASE_INDEXES = {}
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
    boost = 0.0
    if matched_disease and doc.metadata.get("disease") == matched_disease:
        boost += 2.0
    if matched_section and doc.metadata.get("section") == matched_section:
        boost += 1.0
    topic = doc.metadata.get("topic", "")
    if matched_disease and "ho hap" in normalize_text(topic) and "phoi" in normalize_text(matched_disease):
        boost += 0.3
    return boost

def retrieve_hybrid(question: str, top_k: int = RETRIEVER_TOP_K):
    matched_disease, matched_section = infer_query_metadata(question)

    fetch_n = min(RETRIEVER_FETCH_K, len(chunks))
    vector_candidates = vectorstore.similarity_search_with_score(question, k=fetch_n)

    seen: set[str] = set()
    deduped = []
    for doc, orig_dist in vector_candidates:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, float(orig_dist)))

    reranked = sorted(
        deduped,
        key=lambda item: item[1] - metadata_boost(item[0], matched_disease, matched_section),
    )

    top = []
    for doc, orig_dist in reranked[:top_k]:
        boost = metadata_boost(doc, matched_disease, matched_section)
        adjusted = orig_dist - boost
        top.append((doc, adjusted, orig_dist))

    docs = [t[0] for t in top]
    query_meta = {
        "matched_disease": matched_disease,
        "matched_section": matched_section,
        "score_direction": "L2 distance — nhỏ hơn = relevant hơn",
    }
    return docs, top, query_meta

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

class BadFormatStopper(StoppingCriteria):
    STOP_PHRASES = [
        "trắc nghiệm",
        "câu hỏi trắc nghiệm",
        "chọn đáp án",
        "a) ", "b) ", "c) ", "d) ",
    ]

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self._stop_ids = [
            tokenizer.encode(phrase, add_special_tokens=False)
            for phrase in self.STOP_PHRASES
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_tokens = input_ids[0, -30:].tolist()
        decoded = self.tokenizer.decode(last_tokens).lower()
        return any(phrase in decoded for phrase in self.STOP_PHRASES)

def ask_rag(question: str, top_k: int = RETRIEVER_TOP_K, max_new_tokens: int = 400):
    retrieved_docs, reranked_items, query_meta = retrieve_hybrid(question, top_k=top_k)

    context = build_context(retrieved_docs)
    prompt = build_rag_prompt(question, context)

    print(f"Generating answer for question: {question}")
    print(f"Prompt length (chars): {len(prompt)}")
    print(f"Using device: {model.device}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stopper = BadFormatStopper(tokenizer, model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"Answer generated, length {len(generated_tokens)} tokens")

    sources = []
    for doc in retrieved_docs:
        sources.append({
            "disease": doc.metadata.get("disease", ""),
            "section": doc.metadata.get("section", ""),
            "source": doc.metadata.get("source", ""),
            "url": doc.metadata.get("url", ""),
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })

    return answer, sources

def initialize_models():
    global vectorstore, retriever, model, tokenizer, chunks, ALL_DISEASES, DOCS_BY_DISEASE, DISEASE_INDEXES

    print("Loading knowledge base...")
    raw_docs = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            metadata = item.get("metadata", {})
            raw_docs.append(Document(page_content=item["page_content"], metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    print("Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Loading FAISS index...")
    if os.path.exists(FAISS_DIR):
        vectorstore = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(FAISS_DIR)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_TOP_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA_MULT,
        },
    )

    ALL_DISEASES = sorted(
        {doc.metadata.get("disease", "") for doc in chunks if doc.metadata.get("disease", "")},
        key=lambda d: len(normalize_text(d)),
        reverse=True,
    )

    DOCS_BY_DISEASE = {
        disease: [doc for doc in chunks if doc.metadata.get("disease") == disease]
        for disease in ALL_DISEASES
    }

    DISEASE_INDEXES = {
        disease: FAISS.from_documents(docs, embedding_model)
        for disease, docs in DOCS_BY_DISEASE.items()
        if docs
    }

    print("Loading LLM...")
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

    print("Initialization complete!")

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    if model is None or vectorstore is None:
        initialize_models()
    yield

app = FastAPI(
    title="Medical AI RAG API",
    description="API for medical question answering using RAG",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    print(f"POST /ask received question={request.question}")
    try:
        answer, sources = ask_rag(request.question)
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask", response_model=AnswerResponse)
async def ask_question_get(question: str | None = None):
    print(f"GET /ask received question={question}")
    if not question:
        raise HTTPException(
            status_code=405,
            detail="Use POST /ask with JSON body {'question': '...'} or GET /ask?question=...",
        )
    try:
        answer, sources = ask_rag(question)
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics_summary")
async def metrics_summary():
    candidates = [
        os.path.join(PROJECT_DIR, "evaluation", "metrics_summary.json"),
        os.path.join(PROJECT_DIR, "evaluation", "evaluation", "metrics_summary.json"),
    ]
    metrics_path = next((p for p in candidates if os.path.exists(p)), None)
    if metrics_path is None:
        raise HTTPException(status_code=404, detail="Metrics summary not found")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"metrics": data}

if __name__ == "__main__":
    initialize_models()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)