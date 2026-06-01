"""
Few-Shot k=3 Karşılaştırması — KAP-FinQA-TR
============================================
Colab'da kullanım:
    from few_shot_eval_k3 import run_few_shot_eval

    results = run_few_shot_eval()
    # ya da parametrelerle:
    results = run_few_shot_eval(
        train_csv="train.csv",
        test_csv="test.csv",
        n_samples=100,
        k=3,
        copy_to_drive=True,
    )
"""

import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Sabitler ──────────────────────────────────────────────────────────────────
MODELS = [
    ("SmolLM2-360M",   "finetuned_SmolLM2-360M",   "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("TinyLlama-1.1B", "finetuned_TinyLlama-1.1B",  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Qwen2.5-1.5B",   "finetuned_Qwen2.5-1.5B",    "Qwen/Qwen2.5-1.5B-Instruct"),
]

MAX_NEW_TOKENS = 64
MAX_INPUT_LEN  = 768  # few-shot prompt daha uzun


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _bnb_cfg() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _avg_rouge(preds: list[str], refs: list[str]) -> dict:
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1, rL = [], []
    for p, r in zip(preds, refs):
        s = sc.score(str(r), str(p))
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(sum(r1) / len(r1), 4),
        "rougeL": round(sum(rL) / len(rL), 4),
    }


def build_tfidf_retriever(train_df: pd.DataFrame, k: int = 3):
    """TF-IDF vektörizer + kapalı-üzerinde k-shot seçici döndürür."""
    train_texts = (
        train_df["soru"].fillna("") + " " + train_df["baglam"].fillna("")
    ).tolist()
    vectorizer  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_vecs  = vectorizer.fit_transform(train_texts)

    def retrieve(query_soru: str, query_baglam: str) -> list[dict]:
        q_vec = vectorizer.transform([query_soru + " " + query_baglam])
        sims  = cosine_similarity(q_vec, train_vecs).flatten()
        top_k = np.argsort(sims)[::-1][:k]
        return [
            {
                "baglam": str(train_df.iloc[idx].get("baglam", ""))[:250],
                "soru"  : str(train_df.iloc[idx]["soru"]),
                "cevap" : str(train_df.iloc[idx]["cevap"]),
            }
            for idx in top_k
        ]

    return retrieve


def build_zero_shot_prompt(baglam: str, soru: str) -> str:
    return (
        f"### Metin\n{baglam[:300]}\n\n"
        f"### Soru\n{soru}\n\n"
        f"### Cevap\n"
    )


def build_few_shot_prompt(baglam: str, soru: str,
                          shots: list[dict]) -> str:
    parts = [
        f"### Örnek {i}\nMetin: {s['baglam']}\nSoru: {s['soru']}\nCevap: {s['cevap']}"
        for i, s in enumerate(shots, 1)
    ]
    few_shot_block = "\n\n".join(parts)
    return (
        f"{few_shot_block}\n\n"
        f"### Metin\n{baglam[:300]}\n\n"
        f"### Soru\n{soru}\n\n"
        f"### Cevap\n"
    )


# ── Ana değerlendirme fonksiyonu ──────────────────────────────────────────────

def run_few_shot_eval(
    train_csv: str   = "train.csv",
    test_csv:  str   = "test.csv",
    n_samples: int   = 100,
    k:         int   = 3,
    copy_to_drive: bool = True,
) -> dict:
    """
    3 model için zero-shot vs few-shot (k=3) ROUGE karşılaştırması yapar.

    Döndürür:
        all_results: {model_name: {zero_shot, few_shot_k3, predictions}}

    Kaydeder:
        few_shot_k3_results.csv
        few_shot_k3_results.json
        few_shot_k3_comparison.png
    """
    # Veri yükleme
    train_df    = pd.read_csv(train_csv, encoding="utf-8-sig")
    test_df     = pd.read_csv(test_csv,  encoding="utf-8-sig")
    test_sample = test_df.head(n_samples).reset_index(drop=True)
    references  = test_sample["cevap"].astype(str).tolist()

    print(f"Train sütunları: {train_df.columns.tolist()}")
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Kullanılan: {n_samples}")

    # TF-IDF retriever
    retrieve = build_tfidf_retriever(train_df, k=k)
    print(f"✓ TF-IDF vektörizer hazır. K={k}")

    bnb        = _bnb_cfg()
    all_results = {}

    for model_name, out_dir, hf_id in MODELS:
        print(f"\n{'='*60}")
        print(f"🔄 Model: {model_name}")
        print(f"{'='*60}")

        tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base  = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, out_dir)
        model.eval()

        def generate(prompt: str) -> str:
            inputs = tok(
                prompt, return_tensors="pt",
                truncation=True, max_length=MAX_INPUT_LEN,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            return tok.decode(out[0][input_len:], skip_special_tokens=True).strip()

        zero_preds, few_preds = [], []
        for i, row in test_sample.iterrows():
            baglam = str(row.get("baglam", ""))
            soru   = str(row["soru"])

            zero_preds.append(generate(build_zero_shot_prompt(baglam, soru)))

            shots = retrieve(soru, baglam)
            few_preds.append(generate(build_few_shot_prompt(baglam, soru, shots)))

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_samples} örnek işlendi...")

        zs_scores = _avg_rouge(zero_preds, references)
        fs_scores = _avg_rouge(few_preds,  references)

        all_results[model_name] = {
            "zero_shot"   : zs_scores,
            "few_shot_k3" : fs_scores,
            "predictions" : {"zero_shot": zero_preds, "few_shot_k3": few_preds},
        }

        delta = round(fs_scores["rouge1"] - zs_scores["rouge1"], 4)
        print(f"  Zero-shot    → ROUGE-1: {zs_scores['rouge1']} | ROUGE-L: {zs_scores['rougeL']}")
        print(f"  Few-shot k={k} → ROUGE-1: {fs_scores['rouge1']} | ROUGE-L: {fs_scores['rougeL']}")
        print(f"  Delta ROUGE-1: {delta:+}")

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    # ── Özet tablo ─────────────────────────────────────────────────────────
    rows = []
    for model_name, res in all_results.items():
        zs = res["zero_shot"]
        fs = res["few_shot_k3"]
        rows.append({
            "Model"        : model_name,
            "ZS ROUGE-1"   : zs["rouge1"],
            f"FS-k{k} ROUGE-1": fs["rouge1"],
            "Δ ROUGE-1"    : round(fs["rouge1"] - zs["rouge1"], 4),
            "ZS ROUGE-L"   : zs["rougeL"],
            f"FS-k{k} ROUGE-L": fs["rougeL"],
            "Δ ROUGE-L"    : round(fs["rougeL"] - zs["rougeL"], 4),
        })

    summary_df = pd.DataFrame(rows)
    print(f"\n📊 Few-Shot k={k} Karşılaştırma Tablosu")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("few_shot_k3_results.csv", index=False, encoding="utf-8-sig")
    print("✓ few_shot_k3_results.csv kaydedildi.")

    # ── Yanlış cevap analizi ───────────────────────────────────────────────
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    print("\n📌 Örnek bazlı analiz (improved / degraded / same):")
    for model_name, res in all_results.items():
        improved = degraded = same = 0
        for ref, zp, fp in zip(
            references,
            res["predictions"]["zero_shot"],
            res["predictions"]["few_shot_k3"],
        ):
            zs_r = sc.score(ref, zp)["rouge1"].fmeasure
            fs_r = sc.score(ref, fp)["rouge1"].fmeasure
            if fs_r > zs_r + 0.01:
                improved += 1
            elif fs_r < zs_r - 0.01:
                degraded += 1
            else:
                same += 1
        total = improved + degraded + same
        print(f"  {model_name}:")
        print(f"    İyileşti : {improved}/{total} ({100*improved/total:.1f}%)")
        print(f"    Kötüleşti: {degraded}/{total} ({100*degraded/total:.1f}%)")
        print(f"    Değişmedi: {same}/{total} ({100*same/total:.1f}%)")

    # ── Grafik ─────────────────────────────────────────────────────────────
    models = summary_df["Model"].tolist()
    zs_r1  = summary_df["ZS ROUGE-1"].tolist()
    fs_r1  = summary_df[f"FS-k{k} ROUGE-1"].tolist()

    x     = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, zs_r1, width, label="Zero-Shot",     color="#4C8BE2")
    bars2 = ax.bar(x + width / 2, fs_r1, width, label=f"Few-Shot k={k}", color="#E25C4C")
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("ROUGE-1 F1")
    ax.set_title(f"Zero-Shot vs Few-Shot k={k} — ROUGE-1 Karşılaştırması")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, max(max(zs_r1), max(fs_r1)) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("few_shot_k3_comparison.png", dpi=150)
    plt.show()
    print("✓ few_shot_k3_comparison.png kaydedildi.")

    # ── JSON kaydet (predictions hariç) ───────────────────────────────────
    save_results = {
        m: {kk: vv for kk, vv in r.items() if kk != "predictions"}
        for m, r in all_results.items()
    }
    with open("few_shot_k3_results.json", "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print("✓ few_shot_k3_results.json kaydedildi.")

    # ── Drive kopyalama ────────────────────────────────────────────────────
    if copy_to_drive:
        import shutil
        target = "/content/drive/MyDrive/kap_finqa/"
        for fname in ["few_shot_k3_results.csv",
                      "few_shot_k3_results.json",
                      "few_shot_k3_comparison.png"]:
            if os.path.exists(fname):
                shutil.copy(fname, target)
                print(f"✓ {fname} → Drive")

    print("\n✅ Tüm modeller tamamlandı.")
    return all_results


if __name__ == "__main__":
    run_few_shot_eval()
