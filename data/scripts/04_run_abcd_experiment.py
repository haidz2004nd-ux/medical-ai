#!/usr/bin/env python3
"""Run 4-config experiment for Medical QA RAG.

Configs:
  A: Base model, No RAG
  B: Base model, With RAG
  C: Fine-tuned model, No RAG
  D: Fine-tuned model, With RAG

Metrics:
  - BLEU-4
  - ROUGE-L
  - BERTScore F1
  - Recall@5 (RAG only)
  - Human eval CSV/Excel for 50 questions
"""

import argparse
import gc
import json
import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score_fn
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)

import nltk

warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

set_seed(42)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_PATH = os.path.join(PROJECT_DIR, "qa", "test_qa.json")
KB_PATH = os.path.join(PROJECT_DIR, "final_knowledge_base.jsonl")
FAISS_DIR = os.path.join(PROJECT_DIR, "embeddings", "faiss_medical_index_enriched")
LORA_DIR = os.path.join(PROJECT_DIR, "models", "qwen2_5_1_5b_medical_lora")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_NEW_TOKENS = 256
RETRIEVER_TOP_K = 5
RETRIEVER_FETCH_K = 30
RETRIEVER_LAMBDA_MULT = 0.7

DISEASE_SYNONYMS = {
    "viem phoi": ["pneumonia", "sung phoi", "nhiem trung phoi"],
    "dai thao duong": ["tieu duong", "diabetes", "diabetes mellitus"],
    "tang huyet ap": ["cao huyet ap", "hypertension"],
    "gout (gut)": ["gout", "gut", "thong phong"],
    "viem hong": ["dau hong", "pharyngitis"],
    "viem da day": ["dau da day", "gastritis"],
    "tao bon": ["constipation"],
    "tieu chay": ["diarrhea"],
    "tram cam": ["depression"],
    "mat ngu": ["insomnia"],
    "dau dau": ["nhuc dau", "headache"],
    "kho tho": ["shortness of breath", "dyspnea"],
    "ho": ["cough"],
    "ho nhieu": ["ho keo dai", "cough"],
}

SECTION_INTENTS = {
    "symptoms": ["trieu chung", "dau hieu", "bieu hien", "nhan biet"],
    "causes": ["nguyen nhan", "tai sao", "do dau"],
    "treatment": ["dieu tri", "chua", "thuoc", "xu tri"],
    "when_to_see_doctor": ["khi nao", "di kham", "gap bac si", "cap cuu"],
    "prevention": ["phong ngua", "phong benh"],
    "definition": ["la gi", "dinh nghia", "tong quan"],
    "risk_groups": ["nguy co", "doi tuong"],
    "care_notes": ["luu y", "cham soc"],
}


def normalize_text(text: str) -> str:
    x = text.lower()
    x = unicodedata.normalize("NFD", x)
    x = "".join(ch for ch in x if unicodedata.category(ch) != "Mn")
    return x.replace("\u0111", "d")


def contains_whole_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase).strip()
    if not normalized_phrase:
        return False
    return re.search(r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)", text) is not None


def get_question(ex):
    q = ex.get("instruction") or ex.get("question", "")
    extra = ex.get("input", "").strip()
    return f"{q}\n\n{extra}" if extra else q


def get_reference(ex):
    return ex.get("output") or ex.get("answer", "")


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    smoother = SmoothingFunction().method4
    refs_tok = [[nltk.word_tokenize(r.lower())] for r in references]
    hyps_tok = [nltk.word_tokenize(h.lower()) for h in hypotheses]
    return corpus_bleu(refs_tok, hyps_tok, smoothing_function=smoother)


def compute_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hypotheses, references)]
    return float(np.mean(scores))


def compute_bertscore(hypotheses: list[str], references: list[str], lang: str = "vi") -> float:
    _, _, F1 = bert_score_fn(hypotheses, references, lang=lang, model_type="xlm-roberta-base", verbose=False)
    return float(F1.mean())


def compute_recall_at_k(retrieved_docs_list: list[list[Document]], references: list[str], k: int = 5) -> float:
    hits = 0
    for docs, ref in zip(retrieved_docs_list, references):
        ref_norm = normalize_text(ref)
        ref_words = ref_norm.split()
        ref_ngrams = set(" ".join(ref_words[i : i + 3]) for i in range(len(ref_words) - 2))
        found = False
        for doc in docs[:k]:
            doc_norm = normalize_text(doc.page_content)
            if any(ng in doc_norm for ng in ref_ngrams):
                found = True
                break
        if found:
            hits += 1
    return hits / len(references) if references else 0.0


def load_test_data(path: str, max_samples: int | None = 50) -> tuple[list[str], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    questions = [get_question(ex) for ex in data]
    references = [get_reference(ex) for ex in data]
    return questions, references


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_rag_pipeline(device: str = "cuda") -> tuple[list[Document], FAISS]:
    raw_docs = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            raw_docs.append(Document(page_content=item["page_content"], metadata=item.get("metadata", {})))

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(raw_docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(FAISS_DIR):
        vectorstore = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(FAISS_DIR)
    return chunks, vectorstore


def get_disease_aliases(disease: str) -> list[str]:
    normalized = normalize_text(disease)
    aliases = [normalized] + DISEASE_SYNONYMS.get(normalized, [])
    return sorted(set(aliases), key=len, reverse=True)


def infer_query_metadata(question: str, all_diseases: list[str]) -> tuple[str | None, str | None]:
    nq = normalize_text(question)
    matched_disease = None
    for d in all_diseases:
        if any(contains_whole_phrase(nq, alias) for alias in get_disease_aliases(d)):
            matched_disease = d
            break
    matched_section = None
    for sec, kws in SECTION_INTENTS.items():
        if any(k in nq for k in kws):
            matched_section = sec
            break
    return matched_disease, matched_section


def metadata_boost(doc: Document, md: str | None, ms: str | None) -> float:
    boost = 0.0
    if md and doc.metadata.get("disease") == md:
        boost += 2.0
    if ms and doc.metadata.get("section") == ms:
        boost += 1.0
    topic = doc.metadata.get("topic", "")
    if md and "ho hap" in normalize_text(topic) and "phoi" in normalize_text(md):
        boost += 0.3
    return boost


def retrieve_hybrid(question: str, chunks: list[Document], vectorstore: FAISS, all_diseases: list[str], top_k: int = RETRIEVER_TOP_K) -> list[Document]:
    md, ms = infer_query_metadata(question, all_diseases)
    fetch_n = min(RETRIEVER_FETCH_K, len(chunks))
    candidates = vectorstore.similarity_search_with_score(question, k=fetch_n)
    seen = set()
    deduped = []
    for doc, dist in candidates:
        key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, float(dist)))
    reranked = sorted(deduped, key=lambda item: item[1] - metadata_boost(item[0], md, ms))
    return [doc for doc, _ in reranked[:top_k]]


def build_context(docs: list[Document]) -> str:
    parts = []
    for idx, doc in enumerate(docs, 1):
        parts.append(
            f"[Tài liệu {idx}]\n"
            f"Bệnh: {doc.metadata.get('disease','')} | Mục: {doc.metadata.get('section','')}\n"
            f"Nội dung: {doc.page_content}"
        )
    return "\n\n".join(parts)

SYSTEM_PLAIN = (
    "Bạn là trợ lý y tế phổ thông. "
    "Trả lời ngắn gọn, dễ hiểu, không chẩn đoán thay bác sĩ."
)

SYSTEM_RAG = (
    "Bạn là trợ lý hỏi đáp y tế. Trả lời bằng tiếng Việt.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. Chỉ dùng thông tin có trong NGỮ CẢNH bên dưới.\n"
    "2. KHÔNG dùng kiến thức bên ngoài NGỮ CẢNH.\n"
    "3. Nếu không có thông tin phù hợp, trả lời: "
    "\"Tôi chưa có đủ thông tin trong tài liệu để trả lời chính xác.\"\n"
    "4. KHÔNG chẩn đoán bệnh. KHÔNG kê đơn thuốc.\n"
    "5. Trả lời ngắn gọn, đúng trọng tâm, dạng văn xuôi hoặc danh sách (-).\n"
    "6. TUYỆT ĐỐI KHÔNG tạo câu hỏi trắc nghiệm, quiz."
)


def build_prompt_no_rag(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PLAIN},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_rag(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class BadFormatStopper(StoppingCriteria):
    STOP_PHRASES = ["trắc nghiệm", "chọn đáp án", "a) ", "b) ", "c) ", "d) "]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0, -30:]).lower()
        return any(phrase in decoded for phrase in self.STOP_PHRASES)


def generate_answer(model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopper = BadFormatStopper(tokenizer)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def load_base_model():
    print("⏳ Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Base model loaded")
    return model


def load_finetuned_model():
    print("⏳ Loading fine-tuned model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    ft = PeftModel.from_pretrained(base, LORA_DIR)
    ft.config.use_cache = True
    ft.eval()
    print("✅ Fine-tuned model loaded")
    return ft


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️  Model unloaded, VRAM freed")


def run_config(config_name: str, model, questions: list[str], references: list[str], use_rag: bool, chunks: list[Document], vectorstore: FAISS, all_diseases: list[str]) -> dict:
    print(f"\n{'='*60}")
    print(f"▶ Config {config_name} | RAG={'Yes' if use_rag else 'No'}")
    print(f"{'='*60}")

    answers = []
    retrieved_docs_list = []
    for question in tqdm(questions, desc=f"Config {config_name}"):
        try:
            if use_rag:
                docs = retrieve_hybrid(question, chunks, vectorstore, all_diseases)
                prompt = build_prompt_rag(question, build_context(docs))
                retrieved_docs_list.append(docs)
            else:
                prompt = build_prompt_no_rag(question)
                retrieved_docs_list.append([])
            answers.append(generate_answer(model, prompt))
        except Exception as e:
            print(f"  ⚠ Error on question: {e}")
            answers.append("")
            retrieved_docs_list.append([])

    bleu = compute_bleu(answers, references)
    rouge_l = compute_rouge_l(answers, references)
    bertscore_f1 = compute_bertscore(answers, references)
    recall_at_5 = compute_recall_at_k(retrieved_docs_list, references, k=RETRIEVER_TOP_K) if use_rag else None

    metrics = {
        "config": config_name,
        "use_rag": use_rag,
        "bleu_4": round(bleu, 4),
        "rouge_l": round(rouge_l, 4),
        "bertscore_f1": round(bertscore_f1, 4),
        "recall_at_5": round(recall_at_5, 4) if recall_at_5 is not None else "N/A",
        "n_samples": len(answers),
    }
    print(f"\n✅ Config {config_name} finished")
    for key, value in metrics.items():
        print(f"   {key:20s}: {value}")
    return {"metrics": metrics, "answers": answers, "retrieved": retrieved_docs_list}


def save_experiment_results(metrics_list: list[dict], answers: dict, questions: list[str], references: list[str]):
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved metrics_summary.csv/json to {OUTPUT_DIR}")

    human_rows = []
    for idx, (q, ref) in enumerate(zip(questions, references)):
        human_rows.append({
            "stt": idx + 1,
            "question": q,
            "reference_answer": ref,
            "answer_A_base_norag": answers["A"].get(idx, ""),
            "answer_B_base_rag": answers["B"].get(idx, ""),
            "answer_C_ft_norag": answers["C"].get(idx, ""),
            "answer_D_ft_rag": answers["D"].get(idx, ""),
            "score_A (1-5)": "",
            "score_B (1-5)": "",
            "score_C (1-5)": "",
            "score_D (1-5)": "",
            "ghi_chu_reviewer": "",
        })
    df_human = pd.DataFrame(human_rows)
    human_csv = os.path.join(OUTPUT_DIR, "human_eval_50_questions.csv")
    human_xlsx = os.path.join(OUTPUT_DIR, "human_eval_50_questions.xlsx")
    df_human.to_csv(human_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(human_xlsx, engine="openpyxl") as writer:
        df_human.to_excel(writer, index=False, sheet_name="Human Eval")
        ws = writer.sheets["Human Eval"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
    print(f"✅ Saved human_eval CSV/XLSX to {OUTPUT_DIR}")


def main(max_eval_samples: int | None):
    questions, references = load_test_data(TEST_PATH, max_eval_samples)
    print(f"Loaded {len(questions)} evaluation questions")

    print("Building RAG pipeline...")
    chunks, vectorstore = build_rag_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")
    all_diseases = sorted({doc.metadata.get("disease", "") for doc in chunks if doc.metadata.get("disease", "")}, key=lambda d: len(normalize_text(d)), reverse=True)

    global tokenizer
    tokenizer = load_tokenizer()
    print("Tokenizer ready")

    model_base = load_base_model()
    result_A = run_config("A (Base, No RAG)", model_base, questions, references, False, chunks, vectorstore, all_diseases)
    result_B = run_config("B (Base, RAG)", model_base, questions, references, True, chunks, vectorstore, all_diseases)
    unload_model(model_base)

    model_ft = load_finetuned_model()
    result_C = run_config("C (Fine-tuned, No RAG)", model_ft, questions, references, False, chunks, vectorstore, all_diseases)
    result_D = run_config("D (Fine-tuned, RAG)", model_ft, questions, references, True, chunks, vectorstore, all_diseases)
    unload_model(model_ft)

    metrics_list = [result_A["metrics"], result_B["metrics"], result_C["metrics"], result_D["metrics"]]
    answers_map = {
        "A": {i: ans for i, ans in enumerate(result_A["answers"])},
        "B": {i: ans for i, ans in enumerate(result_B["answers"])},
        "C": {i: ans for i, ans in enumerate(result_C["answers"])},
        "D": {i: ans for i, ans in enumerate(result_D["answers"])},
    }
    save_experiment_results(metrics_list, answers_map, questions, references)

    print("\nExperiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-config RAG/No-RAG experiment for Medical QA")
    parser.add_argument("--max-eval-samples", type=int, default=50, help="Number of test samples to evaluate (default 50)")
    args = parser.parse_args()
    main(args.max_eval_samples)
