"""
ADIM 4 — RAG Pipeline (Katman 2)
==================================
Rapordaki Tablo 7 bileşenleri:
  - PDF → Metin: PyMuPDF
  - Chunking: RecursiveCharacterTextSplitter (512 token, 50 overlap)
  - Embedding: intfloat/multilingual-e5-small
  - Vektör DB: ChromaDB (in-memory)
  - Retrieval: Top-k cosine similarity
  - Metrik: Recall@3, Recall@5, MRR

Kullanım:
  python 4_rag_pipeline.py --model Qwen2.5-1.5B --pdf_dir faaliyet_raporlari

Colab kurulum:
  !pip install -q chromadb sentence-transformers langchain pymupdf rouge-score
"""

import argparse, json, os, torch
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import fitz  # PyMuPDF
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re


# ── 1. PDF'leri Oku ve Chunk'la ─────────────────────────────────────────────

def pdf_to_chunks(pdf_path: str, chunk_size=512, overlap=50) -> list:
    """PyMuPDF ile PDF oku, 512 token'lık chunk'lara böl"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()

    # Temizle
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'[ \t]{2,}', ' ', full_text)

    # Basit karakter tabanlı chunking (RecursiveCharacterTextSplitter benzeri)
    words = full_text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 100:
            chunks.append({
                "text"  : chunk,
                "source": os.path.basename(pdf_path),
                "id"    : f"{os.path.basename(pdf_path)}_{start}"
            })
        start += chunk_size - overlap
    return chunks


def build_vectordb(pdf_dir: str, collection_name="kap_finqa"):
    """Tüm PDF'leri oku, ChromaDB'ye yükle"""
    print("\n📚 Vektör veritabanı oluşturuluyor...")

    # multilingual-e5-small (rapor Tablo 7)
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")

    client = chromadb.Client()  # in-memory
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name)

    pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
    print(f"  {len(pdf_files)} PDF bulundu")

    all_chunks = []
    for pdf in pdf_files:
        chunks = pdf_to_chunks(str(pdf))
        all_chunks.extend(chunks)

    print(f"  Toplam chunk: {len(all_chunks)}")

    # Batch olarak ekle
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        ids   = [c["id"]   for c in batch]
        metas = [{"source": c["source"]} for c in batch]
        embeddings = embedder.encode(
            ["passage: " + t for t in texts],  # e5 prefix
            normalize_embeddings=True
        ).tolist()
        collection.add(documents=texts, embeddings=embeddings,
                       ids=ids, metadatas=metas)
        print(f"  ✓ {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunk eklendi")

    print("✓ Vektör DB hazır\n")
    return collection, embedder


# ── 2. Retrieval Metrikleri ──────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list, relevant_source: str, k: int) -> int:
    """Doğru kaynaktan bir chunk top-k içinde var mı? 1/0"""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if relevant_source in doc_id:
            return 1
    return 0


def mrr_score(retrieved_ids: list, relevant_source: str) -> float:
    """Mean Reciprocal Rank — doğru chunk'ın sırasının tersi"""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if relevant_source in doc_id:
            return 1.0 / rank
    return 0.0


# ── 3. RAG + Generation ──────────────────────────────────────────────────────

def rag_generate(tok, model, embedder, collection, soru: str, k=5) -> tuple:
    """Soru için ilgili chunk'ları getir, modele ver, cevap üret"""
    query_emb = embedder.encode(
        ["query: " + soru], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas"]
    )

    retrieved_ids  = results["ids"][0]
    retrieved_docs = results["documents"][0]
    context = "\n\n---\n\n".join(retrieved_docs[:3])  # top-3 bağlam

    prompt = (
        "### Görev\n"
        "Aşağıdaki finansal rapor metinlerine dayanarak soruyu kısa ve kesin yanıtla.\n\n"
        f"### Bağlam\n{context[:800]}\n\n"
        f"### Soru\n{soru}\n\n"
        "### Cevap\n"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=64,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )
    pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()
    return pred, retrieved_ids


# ── 4. Ana Akış ──────────────────────────────────────────────────────────────

def main(model_name: str, pdf_dir: str, test_csv: str):
    from pathlib import Path

    # Fine-tuned modeli yükle
    MODEL_MAP = {
        "SmolLM2-360M" : "HuggingFaceTB/SmolLM2-360M-Instruct",
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-1.5B" : "Qwen/Qwen2.5-1.5B-Instruct",
        "Gemma-4-E2B"  : "google/gemma-3-1b-it",
    }
    hf_id   = MODEL_MAP[model_name]
    out_dir = f"finetuned_{model_name.replace('/', '_')}"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base  = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=bnb,
                                                  device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, out_dir)
    model.eval()

    # Vektör DB kur
    collection, embedder = build_vectordb(pdf_dir)

    # Test seti
    df = pd.read_csv(test_csv, encoding="utf-8-sig")

    # Metrikler
    r3_list, r5_list, mrr_list = [], [], []
    rouge1_list, rougeL_list, bleu_list = [], [], []
    results = []
    rscorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    smooth  = SmoothingFunction().method1

    for i, row in enumerate(df.itertuples(), 1):
        soru   = str(row.soru)
        gold   = str(row.cevap)
        source = str(row.kaynak)

        pred, retrieved_ids = rag_generate(tok, model, embedder, collection, soru, k=5)

        # Retrieval metrikleri
        r3  = recall_at_k(retrieved_ids, source, k=3)
        r5  = recall_at_k(retrieved_ids, source, k=5)
        mrr = mrr_score(retrieved_ids, source)
        r3_list.append(r3); r5_list.append(r5); mrr_list.append(mrr)

        # Generation metrikleri
        s = rscorer.score(gold, pred)
        r1   = round(s["rouge1"].fmeasure, 4)
        rl   = round(s["rougeL"].fmeasure, 4)
        bleu = round(sentence_bleu([gold.split()], pred.split(), smoothing_function=smooth), 4)
        rouge1_list.append(r1); rougeL_list.append(rl); bleu_list.append(bleu)

        results.append({
            "soru": soru, "gold": gold, "pred": pred,
            "recall@3": r3, "recall@5": r5, "mrr": mrr,
            "rouge1": r1, "rougeL": rl, "bleu": bleu,
        })

        if i % 10 == 0:
            print(f"  [{i}/{len(df)}] R@3={sum(r3_list)/len(r3_list):.3f} "
                  f"MRR={sum(mrr_list)/len(mrr_list):.3f} "
                  f"ROUGE-1={sum(rouge1_list)/len(rouge1_list):.3f}")

    n = len(results)
    summary = {
        "model"       : model_name,
        "n_test"      : n,
        "Recall@3"    : round(sum(r3_list)/n, 4),
        "Recall@5"    : round(sum(r5_list)/n, 4),
        "MRR"         : round(sum(mrr_list)/n, 4),
        "avg_rouge1"  : round(sum(rouge1_list)/n, 4),
        "avg_rougeL"  : round(sum(rougeL_list)/n, 4),
        "avg_bleu"    : round(sum(bleu_list)/n, 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — RAG Sonuçları")
    for k, v in summary.items():
        if k not in ["model", "n_test"]:
            print(f"  {k:12s}: {v}")
    print(f"{'='*50}")

    out_file = f"rag_eval_{model_name.replace('/', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"✓ Kaydedildi: {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--pdf_dir",  default="faaliyet_raporlari")
    ap.add_argument("--test_csv", default="test.csv")
    args = ap.parse_args()
    main(args.model, args.pdf_dir, args.test_csv)
