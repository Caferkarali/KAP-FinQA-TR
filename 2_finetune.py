"""
ADIM 2 — QLoRA Fine-Tuning
Kullanım:
  python 2_finetune.py --model Qwen2.5-1.5B --train_csv /train.csv --val_csv /validation.csv
"""

import argparse, json, torch, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_MAP = {
    "SmolLM2-360M" : "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B" : "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-4-E2B"  : "google/gemma-3-1b-it",
}

def format_row(row) -> str:
    baglan = str(row.get("baglam", ""))[:300]
    return (
        "### Görev\nAşağıdaki metne dayanarak soruyu kısa yanıtla.\n\n"
        f"### Metin\n{baglan}\n\n"
        f"### Soru\n{row['soru']}\n\n"
        f"### Cevap\n{row['cevap']}"
    )

def main(model_name, train_csv, val_csv, seed):
    hf_id   = MODEL_MAP[model_name]
    out_dir = f"finetuned_{model_name}_seed{seed}"
    print(f"\n🚀 Fine-tuning: {model_name}  |  seed={seed}\n")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=["q_proj","v_proj"],
    ))
    model.print_trainable_parameters()

    train_df = pd.read_csv(train_csv, encoding="utf-8-sig")
    val_df   = pd.read_csv(val_csv,   encoding="utf-8-sig")
    train_df["text"] = train_df.apply(format_row, axis=1)
    val_df["text"]   = val_df.apply(format_row, axis=1)
    train_ds = Dataset.from_pandas(train_df[["text"]])
    val_ds   = Dataset.from_pandas(val_df[["text"]])
    print(f"✓ Train: {len(train_ds)} | Val: {len(val_ds)}")

    def tokenize(sample):
        return tok(sample["text"], truncation=True, max_length=256, padding="max_length")

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])
    train_ds = train_ds.map(lambda x: {"labels": x["input_ids"]})
    val_ds   = val_ds.map(lambda x: {"labels": x["input_ids"]})

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        fp16=False,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\n⏳ Eğitim başlıyor...")
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    with open(f"{out_dir}/train_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Tamamlandı: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True, choices=list(MODEL_MAP.keys()))
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--val_csv",   default="validation.csv")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()
    main(args.model, args.train_csv, args.val_csv, args.seed)
