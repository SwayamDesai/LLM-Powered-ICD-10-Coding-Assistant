"""
ICD-10 Code Classification Pipeline
Author: Swayam Desai
Project: Inference Analytics AI – Capstone
Description:
A modular pipeline that classifies clinical text into ICD-10 codes
using embeddings, vector search, reranking, and a lightweight LLM.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import zipfile
import requests
import pandas as pd
import numpy as np
import re
import torch
from typing import List, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient, DataType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================================
# CONFIGURATION
# ============================================================================
COLLECTION_NAME = "icd10codes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 5
BATCH_SIZE = 512
USE_4BIT = False  # Memory-efficient quantization


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def download_icd10_codes():
    """Download and parse ICD-10 code data from CMS."""
    print("📥 Downloading ICD-10 codes...")
    url = "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip"
    extract_dir = "./icd10_data"
    os.makedirs(extract_dir, exist_ok=True)

    resp = requests.get(url, timeout=30)
    zip_path = os.path.join(extract_dir, "icd10.zip")
    with open(zip_path, "wb") as f:
        f.write(resp.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    txt_path = os.path.join(extract_dir, "icd10cm_codes_2025.txt")
    codes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                codes.append({"code": parts[0], "description": parts[1]})

    print(f"✅ Loaded {len(codes):,} ICD-10 codes")
    return codes


# ============================================================================
# 2. VECTOR DATABASE
# ============================================================================
class VectorDB:
    """Handles embedding and vector search with Milvus."""

    def __init__(self, milvus_uri: str, milvus_token: str):
        self.client = MilvusClient(uri=milvus_uri, token=milvus_token)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Connected to Milvus (embedding dim: {self.dim})")

    def setup(self, codes: List[Dict], force_recreate=False):
        """Create or load Milvus collection."""
        if self.client.has_collection(COLLECTION_NAME) and not force_recreate:
            self.client.load_collection(COLLECTION_NAME)
            return
        if self.client.has_collection(COLLECTION_NAME):
            self.client.drop_collection(COLLECTION_NAME)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("code", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("description", DataType.VARCHAR, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

        descriptions = [c["description"] for c in codes]
        all_embeddings = []
        for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Embedding"):
            batch = descriptions[i:i+BATCH_SIZE]
            embs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(embs.astype(np.float32))

        vectors = np.vstack(all_embeddings)
        records = [
            {
                "code": codes[i]["code"],
                "description": codes[i]["description"],
                "embedding": vectors[i].tolist()
            }
            for i in range(len(codes))
        ]
        self.client.insert(COLLECTION_NAME, records)
        self.client.load_collection(COLLECTION_NAME)
        print("✅ Milvus setup complete.")

    def search(self, query: str, top_k=TOP_K_RETRIEVAL):
        """Search for similar codes."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[q_emb.tolist()],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["code", "description"]
        )
        return [
            {"code": hit["entity"]["code"], "description": hit["entity"]["description"], "score": float(hit["distance"])}
            for hit in results[0]
        ]


# ============================================================================
# 3. RERANKER
# ============================================================================
class Reranker:
    """Re-ranks retrieved ICD-10 candidates using a cross-encoder."""

    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, candidates: List[Dict], top_k=TOP_K_FINAL):
        pairs = [(query, c["description"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ============================================================================
# 4. LOCAL LLM CLASSIFIER
# ============================================================================
class LocalLLMClassifier:
    """Selects final ICD-10 code using a local LLM."""

    def __init__(self):
        bnb_config = BitsAndBytesConfig(load_in_4bit=USE_4BIT) if USE_4BIT else None
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if not USE_4BIT else None
        )

    def classify(self, query: str, candidates: List[Dict]) -> Dict:
        candidates_str = "\n".join(
            [f"{i+1}. {c['code']}: {c['description']}" for i, c in enumerate(candidates)]
        )
        prompt = f"""You are an expert medical coder.
Clinical query: "{query}"
Candidates:
{candidates_str}
Select the most appropriate ICD-10 code and explain briefly.
"""
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"query": query, "response": response}


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================
class ICD10Pipeline:
    def __init__(self, milvus_uri: str, milvus_token: str):
        self.vector_db = VectorDB(milvus_uri, milvus_token)
        self.reranker = Reranker()
        self.llm = LocalLLMClassifier()

    def setup(self, force_recreate=False):
        codes = download_icd10_codes()
        self.vector_db.setup(codes, force_recreate)

    def classify(self, query: str) -> Dict:
        candidates = self.vector_db.search(query)
        reranked = self.reranker.rerank(query, candidates)
        return self.llm.classify(query, reranked)


if __name__ == "__main__":
    MILVUS_URI = "YOUR_MILVUS_URI"
    MILVUS_TOKEN = "YOUR_MILVUS_TOKEN"

    pipeline = ICD10Pipeline(MILVUS_URI, MILVUS_TOKEN)
    pipeline.setup(force_recreate=False)

    test_queries = [
        "Patient with acute chest pain radiating to left arm",
        "Type 2 diabetes with diabetic neuropathy",
    ]

    for query in test_queries:
        result = pipeline.classify(query)
        print(result)
