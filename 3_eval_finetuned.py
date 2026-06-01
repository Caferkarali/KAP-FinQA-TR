"""
ADIM 3 — Fine-Tuning Sonrası Değerlendirme (ROUGE / BLEU)
===========================================================
Fine-tuned modeli test setinde değerlendirir.
Rapordaki Tablo 6'yı tamamlar (fine-tuning sonrası sütunlar).

Kullanım:
  python 3_eval_finetuned.py --model Qwen2.5-1.5B
  python 3_eval_finetuned.py --model TinyLlama-1.1B
  python 3_eval_finetuned.py --model SmolLM2-360M
  python 3_eval_finetuned.py --model Gemma-4-E2B
"""

import argparse, json, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)


MODEL_MAP = {
    "SmolLM2-360M" : "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B" : "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-4-E2B"  : "google/gemma-3-1b-it",
}


def load_finetuned(model_name: str):
    hf_id   = MODEL_MAP[model_name]
    out_dir = f"finetuned_{model_name.replace('/', '_')}"

    print(f"\n📦 Fine-tuned model yükleniyor: {out_dir}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, out_dir)
    model.eval()
    return tok, model


def generate(tok, model, soru: str, baglan: str, max_new=64) -> str:
    prompt = (
        "### Görev\n"
        "Aşağıdaki finansal rapor metnine dayanarak soruyu kısa ve kesin yanıtla.\n\n"
        f"### Metin\n{baglan[:300]}\n\n"
        f"### Soru\n{soru}\n\n"
        "### Cevap\n"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def compute_metrics(pred: str, gold: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    s = scorer.score(gold, pred)
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([gold.split()], pred.split(), smoothing_function=smooth)
    return round(s["rouge1"].fmeasure, 4), round(s["rougeL"].fmeasure, 4), round(bleu, 4)


def main(model_name: str, test_csv: str, seed: int):
    tok, model = load_finetuned(model_name)
    df = pd.read_csv(test_csv, encoding="utf-8-sig")

    results = []
    r1s, rls, bleus = [], [], []

    for i, row in enumerate(df.itertuples(), 1):
        baglan = str(getattr(row, "baglam", ""))
        pred   = generate(tok, model, str(row.soru), baglan)
        r1, rl, bleu = compute_metrics(pred, str(row.cevap))
        r1s.append(r1); rls.append(rl); bleus.append(bleu)

        results.append({
            "soru": row.soru, "gold": row.cevap, "pred": pred,
            "rouge1": r1, "rougeL": rl, "bleu": bleu,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(df)}] R1={sum(r1s)/len(r1s):.3f} RL={sum(rls)/len(rls):.3f}")

    avg = {
        "model"      : model_name,
        "n_test"     : len(results),
        "avg_rouge1" : round(sum(r1s)/len(r1s), 4),
        "avg_rougeL" : round(sum(rls)/len(rls), 4),
        "avg_bleu"   : round(sum(bleus)/len(bleus), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Fine-Tuned Sonuçları")
    print(f"  ROUGE-1 : {avg['avg_rouge1']}")
    print(f"  ROUGE-L : {avg['avg_rougeL']}")
    print(f"  BLEU    : {avg['avg_bleu']}")
    print(f"{'='*50}")

    out_file = f"eval_finetuned_{model_name.replace('/', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": avg, "details": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"✓ Kaydedildi: {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--test_csv", default="test.csv")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()
    main(args.model, args.test_csv, args.seed)
