from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import json
import pandas as pd
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu

nltk.download("punkt")
nltk.download("punkt_tab")

# =============================================================
# GANTI: MODEL INDONESIA YANG VALID
# =============================================================

MODEL_NAME = "indonesian-nlp/gpt2"   # <--- MODEL FIX !!! 

# =============================================================
# LOAD DATASET
# =============================================================

DATAFILE = "python_scripts_id_500.json"

with open(DATAFILE, "r", encoding="utf-8") as f:
    data_json = json.load(f)

df = pd.DataFrame(data_json)

# gabungkan title + script
df["text"] = df.apply(
    lambda row: f"Judul: {row['title']}\nScript: {row['script']}",
    axis=1
)

dataset = Dataset.from_pandas(df[["text"]])
dataset = dataset.train_test_split(test_size=0.1)

# =============================================================
# LOAD TOKENIZER & MODEL
# =============================================================

print("Loading tokenizer & model…")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# =============================================================
# TOKENIZATION
# =============================================================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# =============================================================
# TRAINING SETUP
# =============================================================

training_args = TrainingArguments(
    output_dir="./indo-scriptgen",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="no",
    save_strategy="no",
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=20,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

# =============================================================
# TRAIN
# =============================================================
print("Training model…")
trainer.train()

# =============================================================
# SAVE MODEL
# =============================================================
print("Saving model…")
trainer.save_model("./indo-scriptgen")
tokenizer.save_pretrained("./indo-scriptgen")

# =============================================================
# BLEU OPTIONAL
# =============================================================

print("Evaluating BLEU…")
refs = [[nltk.word_tokenize(t)] for t in df["text"]]
gens = []

for title in df["title"]:
    prompt = f"Judul: {title}\nScript:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}


    output = model.generate(
        **inputs,
        max_length=256,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    gens.append(nltk.word_tokenize(decoded))

bleu = corpus_bleu(refs, gens)
print("BLEU score:", bleu)

print("Training selesai!")
