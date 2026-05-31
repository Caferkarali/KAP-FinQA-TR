## Model Kartı 4 — Gemma-4-E2B (KAP-FinQA-TR)

| Alan | Değer |
|------|-------|
| **Model Adı** | caferkarali/KAP-FinQA-TR-Gemma-4-E2B |
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

**Performans Metrikleri:**

| Metrik | Zero-Shot | Fine-Tuned |
|--------|-----------|------------|
| ROUGE-1 | 0.2419 | 0.2419 |
| ROUGE-L | 0.2254 | 0.2254 |
| Inference | — | 5732.382 ms/örnek |
| Model Boyutu | — | 3036.0 MB |
| VRAM (PyTorch ayrılan) | — | 3.227 GB |
| VRAM (nvidia-smi net) | — | 6.57 GB |

**Bilinen Limitasyonlar:**
- T4 GPU'da 4-bit kuantizasyon zorunludur; vision tower LoRA ağırlıkları eğitilmemiştir.
- 256 token bağlam kırpması uzun finansal tablolarda bilgi kaybına yol açmaktadır.
- Aritmetik hesaplama gerektiren sorularda fine-tuning yeterli değildir.
- Zero-shot ile fine-tuned ROUGE skorları aynı çıkmıştır; bu durum değerlendirme bölümünde tartışılmalıdır.

**Çalıştırma Gereksinimleri:**
- GPU: NVIDIA T4 veya üstü (minimum 8 GB VRAM, 4-bit kuantizasyon ile)
- RAM: 16 GB sistem belleği
- Python ≥ 3.10, transformers ≥ 4.40, peft, bitsandbytes
