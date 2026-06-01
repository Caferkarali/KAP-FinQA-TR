"""
Gemma-3-4B Model Kartı + Tüm Modeller GPU Bellek Tam Tablosu
============================================================
Colab'da kullanım:
    from gemma_gpu_metrics import run_eval, run_gpu_table, run_model_card, run_all

    run_all()                          # her şeyi sırayla çalıştırır
    run_eval()                         # sadece Gemma değerlendirmesi
    run_gpu_table()                    # sadece GPU tablo ölçümü
    run_model_card()                   # sadece model kartı (eval JSON gerekli)
"""

import gc
import json
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import Dataset

# ── Sabitler ──────────────────────────────────────────────────────────────────
GEMMA_HF_ID  = "google/gemma-3-4b-it"
GEMMA_OUT    = "finetuned_Gemma-3-4B"
EVAL_JSON    = "eval_gemma_3_4b.json"
GPU_CSV      = "gpu_memory_full_table.csv"
GPU_PNG      = "gpu_memory_comparison.png"
MODEL_CARD_MD= "gemma_model_card.md"

MODELS_ALL = [
    ("SmolLM2-360M",   "finetuned_SmolLM2-360M",   "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("TinyLlama-1.1B", "finetuned_TinyLlama-1.1B",  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Qwen2.5-1.5B",   "finetuned_Qwen2.5-1.5B",    "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Gemma-3-4B",     "finetuned_Gemma-3-4B",       "google/gemma-3-4b-it"),
]


def _bnb_cfg(dtype=torch.float16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )


def _row_to_text(row) -> str:
    baglam = str(row.get("baglam", ""))[:300]
    return (
        f"### Metin\n{baglam}\n\n"
        f"### Soru\n{row['soru']}\n\n"
        f"### Cevap\n{row['cevap']}"
    )


def _prompt(row) -> str:
    return (
        f"### Metin\n{str(row.get('baglam', ''))[:300]}\n\n"
        f"### Soru\n{row['soru']}\n\n### Cevap\n"
    )


def _avg_rouge(preds, refs):
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1, rL = [], []
    for p, r in zip(preds, refs):
        s = sc.score(str(r), str(p))
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return round(sum(r1) / len(r1), 4), round(sum(rL) / len(rL), 4)


def _get_nvidia_smi_vram() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            encoding="utf-8",
        ).strip()
        used, total, free = [int(x.strip()) for x in out.split(",")]
        return {"used_mb": used, "total_mb": total, "free_mb": free}
    except Exception as e:
        return {"error": str(e)}


# ── 1. Gemma-3-4B Değerlendirme ───────────────────────────────────────────────
def run_eval(test_csv: str = "test.csv", n_samples: int = 200) -> dict:
    """
    Gemma-3-4B zero-shot ve fine-tuned ROUGE skorlarını hesaplar.
    Sonucu eval_gemma_3_4b.json olarak kaydeder ve dict olarak döndürür.
    """
    test_df     = pd.read_csv(test_csv, encoding="utf-8-sig")
    test_sample = test_df.head(n_samples).reset_index(drop=True)
    refs        = test_sample["cevap"].astype(str).tolist()
    bnb         = _bnb_cfg()

    # ── Zero-shot ──────────────────────────────────────────────────────────
    print("⏱ Zero-shot değerlendirmesi...")
    tok_zs  = AutoTokenizer.from_pretrained(GEMMA_HF_ID, trust_remote_code=True)
    if tok_zs.pad_token is None:
        tok_zs.pad_token = tok_zs.eos_token

    base_zs = AutoModelForCausalLM.from_pretrained(
        GEMMA_HF_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True
    )
    base_zs.eval()

    zs_preds = []
    for _, row in test_sample.iterrows():
        inputs = tok_zs(_prompt(row), return_tensors="pt",
                        truncation=True, max_length=512).to(base_zs.device)
        ilen = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = base_zs.generate(**inputs, max_new_tokens=64,
                                   do_sample=False, pad_token_id=tok_zs.eos_token_id)
        zs_preds.append(tok_zs.decode(out[0][ilen:], skip_special_tokens=True).strip())

    del base_zs
    gc.collect()
    torch.cuda.empty_cache()

    # ── Fine-tuned ─────────────────────────────────────────────────────────
    print("⏱ Fine-tuned değerlendirmesi...")
    tok_ft  = AutoTokenizer.from_pretrained(GEMMA_OUT, trust_remote_code=True)
    if tok_ft.pad_token is None:
        tok_ft.pad_token = tok_ft.eos_token

    base_ft  = AutoModelForCausalLM.from_pretrained(
        GEMMA_HF_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True
    )
    ft_model = PeftModel.from_pretrained(base_ft, GEMMA_OUT)
    ft_model.eval()

    vram_gb  = torch.cuda.memory_allocated() / 1024**3
    size_mb  = sum(p.numel() * p.element_size()
                   for p in ft_model.parameters()) / 1024**2

    # Warmup
    wi = tok_ft("test", return_tensors="pt").to(ft_model.device)
    with torch.no_grad():
        for _ in range(3):
            ft_model.generate(**wi, max_new_tokens=5,
                               pad_token_id=tok_ft.eos_token_id)
    torch.cuda.synchronize()

    ft_preds, times = [], []
    for _, row in test_sample.iterrows():
        inputs = tok_ft(_prompt(row), return_tensors="pt",
                        truncation=True, max_length=512).to(ft_model.device)
        ilen = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = ft_model.generate(**inputs, max_new_tokens=64,
                                    do_sample=False, pad_token_id=tok_ft.eos_token_id)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        ft_preds.append(tok_ft.decode(out[0][ilen:], skip_special_tokens=True).strip())

    del ft_model, base_ft
    gc.collect()
    torch.cuda.empty_cache()

    # ── Sonuçlar ───────────────────────────────────────────────────────────
    zs_r1, zs_rL = _avg_rouge(zs_preds, refs)
    ft_r1, ft_rL = _avg_rouge(ft_preds, refs)
    inf_ms       = round(sum(times) / len(times), 3)

    metrics = {
        "zero_shot"   : {"rouge1": zs_r1, "rougeL": zs_rL},
        "fine_tuned"  : {"rouge1": ft_r1, "rougeL": ft_rL},
        "inference_ms": inf_ms,
        "model_mb"    : round(size_mb, 1),
        "vram_gb"     : round(vram_gb, 2),
    }

    print(f"\n📊 Gemma-3-4B Sonuçları")
    print(f"  Zero-shot  → ROUGE-1: {zs_r1} | ROUGE-L: {zs_rL}")
    print(f"  Fine-tuned → ROUGE-1: {ft_r1} | ROUGE-L: {ft_rL}")
    print(f"  Inference  : {inf_ms} ms/örnek")
    print(f"  Model boyutu: {size_mb:.1f} MB | VRAM: {vram_gb:.2f} GB")

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✓ {EVAL_JSON} kaydedildi.")
    return metrics


# ── 2. Tüm Modeller GPU Bellek Tablosu ────────────────────────────────────────
def run_gpu_table() -> list[dict]:
    """
    4 modeli sırayla yükler, nvidia-smi + PyTorch ile VRAM ölçer.
    gpu_memory_full_table.csv ve gpu_memory_comparison.png kaydeder.
    Sonucu list[dict] olarak döndürür.
    """
    bnb = _bnb_cfg()

    torch.cuda.empty_cache()
    baseline = _get_nvidia_smi_vram()
    print(f"Baseline VRAM: {baseline.get('used_mb', '?')} MB")

    gpu_table = []
    for model_name, out_dir, hf_id in MODELS_ALL:
        print(f"\n🔍 Ölçülüyor: {model_name} ...")
        torch.cuda.empty_cache()

        tok   = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base  = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb,
            device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, out_dir)
        model.eval()

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb          = sum(p.numel() * p.element_size()
                               for p in model.parameters()) / 1024**2

        torch.cuda.synchronize()
        py_alloc_gb  = torch.cuda.memory_allocated() / 1024**3
        py_reserv_gb = torch.cuda.memory_reserved()  / 1024**3

        smi         = _get_nvidia_smi_vram()
        net_used_mb = smi.get("used_mb", 0) - baseline.get("used_mb", 0)

        row = {
            "Model"               : model_name,
            "Toplam Parametre"    : f"{total_params/1e6:.0f}M",
            "Eğitilebilir Param"  : f"{trainable_params/1e6:.2f}M",
            "Model Boyutu (MB)"   : round(size_mb, 1),
            "PyTorch Ayrılan (GB)": round(py_alloc_gb, 3),
            "PyTorch Rezerve (GB)": round(py_reserv_gb, 3),
            "SMI Kullanılan (MB)" : smi.get("used_mb", "N/A"),
            "SMI Net Delta (MB)"  : net_used_mb,
            "GPU Toplam (MB)"     : smi.get("total_mb", "N/A"),
        }
        gpu_table.append(row)

        print(f"  Parametre     : {row['Toplam Parametre']} (eğitilebilir: {row['Eğitilebilir Param']})")
        print(f"  Model boyutu  : {row['Model Boyutu (MB)']} MB")
        print(f"  PyTorch VRAM  : {row['PyTorch Ayrılan (GB)']} GB ayrılmış / {row['PyTorch Rezerve (GB)']} GB rezerve")
        print(f"  nvidia-smi    : {row['SMI Kullanılan (MB)']} MB (net delta: {net_used_mb} MB)")

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    # CSV
    gpu_df = pd.DataFrame(gpu_table)
    print("\n📊 Tüm Modeller — GPU Bellek Tam Tablosu")
    print("=" * 90)
    print(gpu_df.to_string(index=False))
    gpu_df.to_csv(GPU_CSV, index=False, encoding="utf-8-sig")
    print(f"✓ {GPU_CSV} kaydedildi.")

    # Grafik
    models     = [r["Model"] for r in gpu_table]
    py_alloc   = [r["PyTorch Ayrılan (GB)"] for r in gpu_table]
    py_reserve = [r["PyTorch Rezerve (GB)"] for r in gpu_table]
    smi_net_gb = [r["SMI Net Delta (MB)"] / 1024 for r in gpu_table]
    size_list  = [r["Model Boyutu (MB)"] for r in gpu_table]

    x     = np.arange(len(models))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    b1 = ax.bar(x - width, py_alloc,   width, label="PyTorch Ayrılan",  color="#4C8BE2")
    b2 = ax.bar(x,         py_reserve, width, label="PyTorch Rezerve",  color="#A0C4FF")
    b3 = ax.bar(x + width, smi_net_gb, width, label="nvidia-smi Net",   color="#E25C4C")
    for b in list(b1) + list(b2) + list(b3):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("VRAM Kullanımı (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_ylabel("GB")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax2    = axes[1]
    colors = ["#52B788", "#74C69D", "#95D5B2", "#B7E4C7"]
    bars   = ax2.bar(models, size_list, color=colors)
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                 f"{b.get_height():.0f} MB", ha="center", va="bottom", fontsize=8)
    ax2.set_title("Model Disk Boyutu (MB)")
    ax2.set_ylabel("MB")
    ax2.set_xticklabels(models, rotation=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("KAP-FinQA-TR — Tüm Modeller GPU & Boyut Karşılaştırması", fontsize=12)
    plt.tight_layout()
    plt.savefig(GPU_PNG, dpi=150)
    plt.show()
    print(f"✓ {GPU_PNG} kaydedildi.")
    return gpu_table


# ── 3. Gemma-3-4B Model Kartı ─────────────────────────────────────────────────
def run_model_card(gpu_table: list[dict] | None = None) -> str:
    """
    eval_gemma_3_4b.json dosyasından metrikleri okuyarak model kartı oluşturur.
    gemma_model_card.md kaydeder ve kart metnini döndürür.
    gpu_table: run_gpu_table() çıktısı (opsiyonel, yoksa '—' kullanılır)
    """
    try:
        with open(EVAL_JSON, encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {
            "zero_shot"   : {"rouge1": 0.0, "rougeL": 0.0},
            "fine_tuned"  : {"rouge1": 0.0, "rougeL": 0.0},
            "inference_ms": 0.0,
            "model_mb"    : 0,
            "vram_gb"     : 0,
        }
        print(f"⚠️  {EVAL_JSON} bulunamadı, placeholder değerler kullanılıyor.")

    gemma_gpu = {"PyTorch Ayrılan (GB)": "—", "SMI Net Delta (MB)": "—"}
    if gpu_table:
        try:
            gemma_gpu = next(r for r in gpu_table if "Gemma" in r["Model"])
        except StopIteration:
            pass

    net_mb = gemma_gpu.get("SMI Net Delta (MB)", "—")
    net_gb = (round(net_mb / 1024, 2)
              if isinstance(net_mb, (int, float)) else "—")

    card = f"""## Model Kartı 4 — Gemma-3-4B (KAP-FinQA-TR)

| Alan | Değer |
|------|-------|
| **Model Adı** | caferkarali/KAP-FinQA-TR-Gemma-3-4B |
| **Temel Model** | google/gemma-3-4b-it |
| **Görev** | Türkçe Finansal Soru-Cevap (Factual Extraction) |
| **Fine-Tuning Yöntemi** | QLoRA (4-bit NF4, r=8, α=16) |
| **Eğitim Verisi** | KAP-FinQA-TR v1.0 — 2.642 örnek |
| **Veri Kaynağı** | KAP (kap.org.tr) — BIST100 faaliyet raporları |
| **Epoch** | 3 |
| **Learning Rate** | 2e-4 (cosine decay) |
| **Batch Size** | 2 (gradient accumulation: 4) |
| **Max Sequence Length** | 256 token |
| **Seed** | 42 |
| **Eğitim Süresi** | ~6-8 saat (Google Colab T4 GPU, tahmini) |

**Performans Metrikleri:**

| Metrik | Zero-Shot | Fine-Tuned |
|--------|-----------|------------|
| ROUGE-1 | {metrics['zero_shot']['rouge1']} | {metrics['fine_tuned']['rouge1']} |
| ROUGE-L | {metrics['zero_shot']['rougeL']} | {metrics['fine_tuned']['rougeL']} |
| Inference | — | {metrics['inference_ms']} ms/örnek |
| Model Boyutu | — | {metrics['model_mb']} MB |
| VRAM (PyTorch allocated) | — | {metrics['vram_gb']} GB |
| VRAM (nvidia-smi net delta) | — | {net_gb} GB |

**Bilinen Limitasyonlar:**
- 4B parametre ile T4 GPU'da 4-bit kuantizasyon zorunludur; yarı hassasiyette A100/V100 gereklidir.
- 256 token bağlam kırpması uzun finansal tablolarda bilgi kaybına yol açmaktadır.
- Aritmetik hesaplama gerektiren sorularda fine-tuning yeterli değildir.
- Çıkarım süresi diğer modellere göre belirgin biçimde yüksektir.

**Çalıştırma Gereksinimleri:**
- GPU: NVIDIA T4 veya üstü (minimum 8 GB VRAM, 4-bit kuantizasyon ile)
- RAM: 16 GB sistem belleği
- Python ≥ 3.10, transformers ≥ 4.40, peft, bitsandbytes
- HuggingFace token (Gemma lisansı için huggingface.co/google/gemma-3-4b-it)
"""

    with open(MODEL_CARD_MD, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"✓ {MODEL_CARD_MD} kaydedildi.")
    print(card)
    return card


# ── 4. Hepsini Sırayla Çalıştır ───────────────────────────────────────────────
def run_all(test_csv: str = "test.csv", n_samples: int = 200,
            copy_to_drive: bool = True):
    """
    Sırasıyla: eval → gpu_table → model_card → (opsiyonel) Drive kopyalama.
    """
    metrics   = run_eval(test_csv=test_csv, n_samples=n_samples)
    gpu_table = run_gpu_table()
    run_model_card(gpu_table=gpu_table)

    if copy_to_drive:
        import shutil
        target = "/content/drive/MyDrive/kap_finqa/"
        files  = [EVAL_JSON, GPU_CSV, GPU_PNG, MODEL_CARD_MD]
        for fname in files:
            if os.path.exists(fname):
                shutil.copy(fname, target)
                print(f"✓ {fname} → Drive")
            else:
                print(f"⚠️  {fname} bulunamadı, atlandı.")
    print("\n🎉 Tamamlandı.")


if __name__ == "__main__":
    run_all()
