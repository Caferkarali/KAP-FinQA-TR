"""
ADIM 1 — Zero-Shot Baseline Değerlendirme
==========================================
Colab'da çalıştır. Fine-tuning öncesi her modelin ham performansını ölçer.
Rapordaki Tablo 6'yı doldurur.

Kullanım:
  python 1_zero_shot_eval.py --model Qwen2.5-1.5B
  python 1_zero_shot_eval.py --model TinyLlama-1.1B
  python 1_zero_shot_eval.py --model SmolLM2-360M
  python 1_zero_shot_eval.py --model Gemma-4-E2B
  python 1_zero_shot_eval.py --model Llama-3-8B
"""

import argparse, json, random, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

# ── Model isimlerini HuggingFace path'ine eşle ──────────────────────────────
MODEL_MAP = {
    "SmolLM2-360M" : "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B" : "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-4-E2B"  : "google/gemma-3-1b-it",   # E2B henüz HF'de yok, 1B ile başla
    "Llama-3-8B"   : "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral-7B"   : "mistralai/Mistral-7B-Instruct-v0.3",
}

def load_model(model_name: str):
    hf_id = MODEL_MAP[model_name]
    print(f"\n📦 Yükleniyor: {hf_id}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def make_prompt(soru: str, baglan: str) -> str:
    return (
        "Aşağıdaki finansal rapor metnine dayanarak soruyu kısa ve kesin olarak yanıtla.\n\n"
        f"METİN:\n{baglan}\n\n"
        f"SORU: {soru}\n\n"
        "CEVAP:"
    )


def generate(tok, model, prompt: str, max_new=64) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def compute_metrics(pred: str, gold: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = scorer.score(gold, pred)
    r1 = round(scores["rouge1"].fmeasure, 4)
    rl = round(scores["rougeL"].fmeasure, 4)

    smooth = SmoothingFunction().method1
    ref = [gold.split()]
    hyp = pred.split()
    bleu = round(sentence_bleu(ref, hyp, smoothing_function=smooth), 4)
    return r1, rl, bleu


def main(model_name: str, test_csv: str, n_samples: int, seed: int):
    random.seed(seed)
    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    # Rastgele n_samples örnek seç
    samples = df.sample(n=min(n_samples, len(df)), random_state=seed)

    tok, model = load_model(model_name)

    results = []
    r1_list, rl_list, bleu_list = [], [], []

    for i, row in enumerate(samples.itertuples(), 1):
        baglan = str(getattr(row, "baglam", ""))[:400]
        soru   = str(row.soru)
        gold   = str(row.cevap)

        prompt = make_prompt(soru, baglan)
        pred   = generate(tok, model, prompt)

        r1, rl, bleu = compute_metrics(pred, gold)
        r1_list.append(r1); rl_list.append(rl); bleu_list.append(bleu)

        results.append({
            "soru": soru, "gold": gold, "pred": pred,
            "rouge1": r1, "rougeL": rl, "bleu": bleu
        })
        print(f"  [{i:02d}/{n_samples}] R1={r1:.3f} RL={rl:.3f} BLEU={bleu:.3f}  |  {soru[:60]}")

    avg = {
        "model"        : model_name,
        "n_samples"    : len(results),
        "avg_rouge1"   : round(sum(r1_list)/len(r1_list), 4),
        "avg_rougeL"   : round(sum(rl_list)/len(rl_list), 4),
        "avg_bleu"     : round(sum(bleu_list)/len(bleu_list), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Zero-Shot Sonuçları")
    print(f"  ROUGE-1 : {avg['avg_rouge1']}")
    print(f"  ROUGE-L : {avg['avg_rougeL']}")
    print(f"  BLEU    : {avg['avg_bleu']}")
    print(f"{'='*50}")

    out_file = f"zero_shot_{model_name.replace('/', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": avg, "details": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"✓ Kaydedildi: {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, choices=list(MODEL_MAP.keys()))
    ap.add_argument("--test_csv", default="test.csv")
    ap.add_argument("--samples",  type=int, default=50)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()
    main(args.model, args.test_csv, args.samples, args.seed)
