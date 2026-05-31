# KAP-FinQA-TR: Küçük Dil Modelleri (SLM) ile Çevrimdışı Finansal RAG 🇹🇷📊

Bu depo, Türkçe finansal raporlar (KAP - Kamuyu Aydınlatma Platformu) üzerinde Küçük Dil Modellerinin (SLM - Small Language Models) QLoRA ile ince ayar (fine-tuning) süreçlerini, RAG (Retrieval-Augmented Generation) performanslarını ve donanım verimliliklerini inceleyen akademik bir benchmark projesini içermektedir.

Bu proje, Selçuk Üniversitesi Bilgisayar Mühendisliği Yüksek Lisans programı "Büyük Dil Modelleri" dersi kapsamında **Cafer Karalı** tarafından geliştirilmiştir. *"Small Language Models for Turkish NLP Tasks: A Comprehensive Fine-Tuning Benchmark"* makalesine katkı sunmak amacıyla standartlaştırılmış protokollere uygun olarak tasarlanmıştır.

---

## 🎯 Proje Motivasyonu ve Kapsamı
KVKK ve BDDK regülasyonları, hassas finansal verilerin bulut tabanlı dış API'ler (GPT-4, Claude vb.) üzerinden işlenmesini sınırlamaktadır. Bu proje, **tamamen yerel (on-premise) ve çevrimdışı** çalışabilen, düşük donanım gereksinimli (Edge AI uyumlu) SLM'lerin kurumsal finansal analizlerde (Factual Extraction) kullanılabilirliğini kanıtlamayı amaçlamaktadır.

## 📊 Değerlendirilen Modeller
Ortak benchmark protokolü gereği aşağıdaki modeller test edilmiş ve karşılaştırılmıştır:
* **SmolLM2-360M** (Edge/Mobil ve CPU deployment uygunluk testi)
* **TinyLlama-1.1B** (Genel amaçlı alt sınır)
* **Qwen2.5-1.5B** (Çok dilli, 32k bağlam pencereli güçlü aday)
* **Gemma-4-E2B** (`google/gemma-3-4b-it` tabanlı MoE mimarisi: ~4B toplam, ~2B aktif parametre)
* **Mistral-7B** *(Yalnızca Zero-Shot üst sınır referansı olarak kullanılmıştır)*

---

## 📂 Depo Yapısı ve Notebook'lar

Proje kodları, modülerlik sağlamak ve Google Colab GPU (T4 16GB) limitlerini optimize etmek amacıyla 4 temel Jupyter Notebook dosyasına bölünmüştür:

* `01_main_pipeline_qwen_tinyllama.ipynb`
  * Qwen2.5-1.5B ve TinyLlama-1.1B modelleri için Zero-Shot değerlendirme, QLoRA Fine-Tuning (seed=42) ve Fine-Tuned (İnce Ayarlı) değerlendirme adımlarını içerir.
* `02_smollm_gemma_pipeline.ipynb`
  * SmolLM2-360M ve Gemma-4-E2B modelleri için eğitim ve değerlendirme süreçleri.
  * ⚠️ *Not: Gemma modeli için Hugging Face üzerinden lisans onayı gereklidir. Notebook içinde HF Token girişi adımı bulunmaktadır.*
* `03_multiseed_reproducibility.ipynb`
  * Akademik tekrarlanabilirlik (reproducibility) kuralı gereği, modellerin `seed=0` ve `seed=123` ile yeniden çalıştırılmasını ve Ortalama ± Standart Sapma metriklerinin hesaplanmasını sağlar.
* `04_rag_and_results.ipynb`
  * BIST100 faaliyet raporları üzerinde RAG mimarisi testleri (PyMuPDF + multilingual-e5-small + ChromaDB).
  * Tüm JSON sonuçlarının okunarak standart benchmark tablosunun oluşturulması ve ROUGE / Verimlilik grafiklerinin (Matplotlib) çizdirilmesi adımlarını barındırır.

### Yardımcı Scriptler (Scripts)
Aşağıdaki Python dosyaları, notebook'lar tarafından çağrılan ana işlevleri barındırır:
* `1_zero_shot_eval.py`: Önceden eğitilmiş (pre-trained) modellerin temel performansını ölçer.
* `2_finetune.py`: QLoRA hiperparametreleri ile model eğitimini gerçekleştirir.
* `3_eval_finetuned.py`: İnce ayar yapılmış modelleri test veri seti üzerinde değerlendirir.
* `4_rag_pipeline.py`: Vektör veritabanı oluşturma ve geri getirme (retrieval) süreçlerini yönetir.
* `5_upload_hf.py`: Eğitilen modelleri Hugging Face Hub'a yükler.

---

## 🚀 Kurulum ve Çalıştırma

Projeyi Google Colab üzerinde çalıştırmak için tasarlanmıştır.

**1. Depoyu Klonlayın ve Drive'a Taşıyın:**
Google Drive'ınızda `kap_finqa` adında bir klasör oluşturun ve repo içeriklerini buraya kopyalayın. Notebook'lar `/content/drive/MyDrive/kap_finqa/` dizini üzerinden çalışacak şekilde ayarlanmıştır.

**2. Gerekli Kütüphaneleri Yükleyin (`requirements.txt` içeriği):**
```bash
pip install transformers peft bitsandbytes trl datasets accelerate rouge-score nltk chromadb sentence-transformers huggingface_hub pymupdf matplotlib

3. Veri Setini Hazırlayın:

Veri seti (train.csv, validation.csv, test.csv) ve test edilecek PDF faaliyet raporlarını kap_finqa/ dizinine yerleştirin.

Notebook'ları sıralı olarak (01 -> 02 -> 03 -> 04) çalıştırın.

🔬 Öne Çıkan Akademik Bulgular
RAG ve Bağlam Kayması (Context Distraction): multilingual-e5-small embedding modeli karmaşık finansal tablo yapılarını ayırt etmekte zorlandığından, küçük dil modelleri (SLM) yanlış/ilgisiz bağlamla beslendiğinde halüsinasyon oranları artmış ve RAG performansı, saf Fine-Tuning performansının ciddi şekilde gerisinde kalmıştır.

Few-Shot Dezavantajı ("Lost in the Middle"): Dar bağlam pencerelerinde (örn. 256 token), k=3 few-shot örneklemi vermek küçük modellerin (SmolLM2 ve TinyLlama) asıl soruyu unutmasına neden olmuş ve performansları Zero-Shot'ın bile altına düşmüştür.

Parametre Başına Verimlilik: SmolLM2-360M, 243 MB disk ayak izi ve CPU'da çalışabilme kapasitesiyle on-premise edge dağıtımı için en verimli model olurken; mutlak doğruluk ve Factual Extraction başarısında en yüksek skoru Qwen2.5-1.5B elde etmiştir.

👨‍💻 Lisans ve İletişim
Bu proje MIT Lisansı altında açık kaynak olarak paylaşılmıştır. (KAP veri setinin kullanımı KAP'ın kendi kullanım koşullarına tabidir).

Geliştirici: Cafer Karalı
