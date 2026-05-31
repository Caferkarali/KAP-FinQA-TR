---
language: tr
tags:
- financial
- turkish
- question-answering
- qlora
- fine-tuned
license: apache-2.0
---

# KAP-FinQA-TR — TinyLlama-1.1B

Türkçe finansal soru-cevap için QLoRA ile fine-tune edilmiş model.

## Proje
Selçuk Üniversitesi Fen Bilimleri Enstitüsü — Büyük Dil Modelleri Dersi Dönem Projesi  
**Küçük Dil Modelleri (SLM) ile Çevrimdışı Finansal RAG**

## Eğitim Detayları
- **Veri Seti**: KAP-FinQA-TR v1.0 (KAP — kap.org.tr)
- **Yöntem**: QLoRA (4-bit NF4, r=8, α=16)
- **Epoch**: 3
- **Seed**: 42

## Kullanım
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("caferkarali/KAP-FinQA-TR-TinyLlama-1.1B")
# base modeli yükle ve adapter ekle
```
