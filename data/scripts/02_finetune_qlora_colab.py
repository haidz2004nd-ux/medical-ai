"""
Fine-tune LLM cho domain y te pho thong bang QLoRA tren Colab Free.

Chay tren Colab:
1. Upload du an len Google Drive.
2. Sua PROJECT_DIR neu duong dan Drive khac.
3. Runtime > Change runtime type > T4 GPU.
4. Chay file nay theo tung cell hoac copy vao notebook.
"""

# %% Cai thu vien
# !pip install -q transformers datasets peft accelerate bitsandbytes

# %% Mount Google Drive
# from google.colab import drive
# drive.mount("/content/drive")

# %% Cau hinh duong dan
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_DIR = "/content/drive/MyDrive/NLP/newdata/data"
TRAIN_PATH = os.path.join(PROJECT_DIR, "qa", "train_qa.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "qwen2_5_1_5b_medical_lora")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
SEED = 42

set_seed(SEED)


# %% Load tokenizer truoc khi format data de dung chat template cua Qwen
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# %% Load QA train data
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

print("So cap QA fine-tune:", len(qa_data))
print("Vi du:", qa_data[0])


def get_question(example):
    question = example.get("instruction") or example.get("question")
    extra_input = example.get("input", "").strip()
    if extra_input:
        return f"{question}\n\n{extra_input}"
    return question


def get_answer(example):
    return example.get("output") or example.get("answer")


def format_example(example):
    question = get_question(example)
    answer = get_answer(example)

    messages = [
        {
            "role": "system",
            "content": (
                "Ban la tro ly y te pho thong. "
                "Hay tra loi ngan gon, de hieu, "
                "khong chan doan thay bac si."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    text += tokenizer.eos_token

    return {"text": text}


dataset = Dataset.from_list(qa_data).map(format_example)
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)


# %% Load model 4-bit cho QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()


# %% Cau hinh LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def tokenize_example(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )


tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    save_total_limit=2,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    data_collator=data_collator,
)


# %% Train va luu adapter
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Da luu LoRA adapter tai:", OUTPUT_DIR)


# %% Test nhanh model fine-tuned
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
ft_model.config.use_cache = True


def ask_finetuned(question, max_new_tokens=256):
    messages = [
        {
            "role": "system",
            "content": (
                "Ban la tro ly y te pho thong. "
                "Hay tra loi ngan gon, de hieu, "
                "khong chan doan thay bac si."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(ft_model.device)
    outputs = ft_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


print(ask_finetuned("Khi nao nguoi bi sot can di kham?"))
