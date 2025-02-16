#!/usr/bin/env python3

"""
Long Context Answer Agent
================================

Key points:
  - We specify `ndim=384` when constructing the USearch index.
  - We call `index.add(label, vector)` with the label first, vector second.
  - Each vector is forced to `float32` and contiguous.

Requirements:
  pip install --upgrade \
      orjson polars zstandard cryptography simplejson python-dotenv openai numpy requests usearch transformers
"""

import os
import time
import math
import datetime as dt
import json
import requests
import orjson
import polars as pl
import numpy as np

from dotenv import load_dotenv
load_dotenv(dotenv_path="keys.env")

from cryptography.fernet import Fernet
import zstandard as zstd

import openai
from transformers import AutoTokenizer
from usearch.index import Index

# -------------------------------------------------------------------
# Tokenizer setup (for truncating texts to <= 50 tokens).
# -------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def truncate_text(text: str, max_tokens=50) -> str:
    encoding = tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
    return tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)


# -------------------------------------------------------------------
# Finweb slow search
# -------------------------------------------------------------------
def finweb_slow_search(
    question: str,
    expansions: list,
    sql_filter: str,
    n_results: int = 1000,
    n_probes: int = 300,
    n_contextify: int = 512,
    algorithm: str = "hybrid-1",
    attempts: int = 0,
) -> pl.DataFrame:
    if attempts >= 3:
        print(f"{dt.datetime.utcnow()} the search has failed three times.")
        return None

    response = requests.post(
        url="https://www.nosible.ai/search/v1/slow-search",
        json={
            "question": question,
            "expansions": expansions,
            "sql_filter": sql_filter,
            "n_results": n_results,
            "n_probes": n_probes,
            "n_contextify": n_contextify,
            "algorithm": algorithm,
        },
        headers={
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json",
            "api-key": os.environ["FINWEB_V1_API_KEY"],
        },
    )

    if response.status_code == 200:
        print(f"{dt.datetime.utcnow()} slow search was accepted.")
        details = response.json()
        download_link = details["download_from"]
        decrypt_key = details["decrypt_using"]
        print("\nDOWNLOAD LINK:", download_link)
        print("DECRYPTION KEY:", decrypt_key, "\n")

        for i in range(10):
            print(f"{dt.datetime.utcnow()} checking the download link.")
            r = requests.get(download_link)
            if r.ok:
                print(f"{dt.datetime.utcnow()} decrypting the data.")
                dec = Fernet(decrypt_key.encode("utf-8")).decrypt(r.content)

                print(f"{dt.datetime.utcnow()} decompressing the data.")
                decmp = zstd.ZstdDecompressor().decompress(dec)

                print(f"{dt.datetime.utcnow()} deserializing the data.")
                obj = orjson.loads(decmp)

                print(f"{dt.datetime.utcnow()} extracting search results.")
                recs = []
                for item in obj["response"]:
                    content = item["content"]
                    if not isinstance(content, str):
                        import simplejson
                        content = simplejson.dumps(content)
                    recs.append(
                        {
                            "question": question,
                            "url_hash": item["url_hash"],
                            "url": item["url"],
                            "title": item["title"],
                            "content": content,
                            "similarity": item["semantics"]["similarity"],
                        }
                    )
                return pl.from_dicts(recs)
            else:
                print(f"{dt.datetime.utcnow()} sleeping for 30s.")
                time.sleep(30)

        # If not successful:
        print(f"{dt.datetime.utcnow()} resubmitting slow-search.")
        return finweb_slow_search(
            question, expansions, sql_filter, n_results, n_probes, n_contextify, algorithm, attempts + 1
        )
    else:
        print(f"Error {response.status_code}: {response.text}")


# -------------------------------------------------------------------
# Helper to parse embedding from HF
# -------------------------------------------------------------------
def extract_embedding(item):
    if isinstance(item, list):
        if all(isinstance(x, (int, float)) for x in item):
            return item
        for sub in item:
            c = extract_embedding(sub)
            if c is not None:
                return c
    elif isinstance(item, dict):
        for k in ["embedding", "embeddings", "data"]:
            if k in item:
                c = extract_embedding(item[k])
                if c is not None:
                    return c
        for val in item.values():
            c = extract_embedding(val)
            if c is not None:
                return c
    return None


# -------------------------------------------------------------------
# HF embedding
# -------------------------------------------------------------------
def remotely_embed_sentences_sync(sentences: list) -> np.ndarray:
    if len(sentences) > 16:
        raise ValueError("Up to 16 sentences per call.")

    # truncate
    s2 = [truncate_text(s, max_tokens=50) for s in sentences]

    endpoint = "https://j3rdbna8u6w2m9zq.us-east-1.aws.endpoints.huggingface.cloud"
    r = requests.post(
        url=endpoint,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"inputs": s2},
        timeout=(5, 15),
    )

    data = json.loads(r.content)
    items = []
    if isinstance(data, list):
        for x in data:
            c = extract_embedding(x)
            if c is None:
                raise ValueError(f"Couldn't parse embedding from: {x}")
            items.append(c)
    elif isinstance(data, dict):
        c = extract_embedding(data)
        if c is None:
            raise ValueError(f"Couldn't parse embedding from: {data}")
        items.append(c)
    else:
        raise ValueError(f"Unexpected embedding response: {data}")
    return np.array(items, dtype=np.float32)


# -------------------------------------------------------------------
# OpenRouter generation
# -------------------------------------------------------------------
import openai

def generate(prompt: str) -> str:
    cli = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["ROUTER_API_KEY"],
    )
    resp = cli.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return resp.choices[0].message.content


# -------------------------------------------------------------------
# USearch index
# -------------------------------------------------------------------
class LongContextAgent:
    def __init__(self):
        self.search_results_df = None
        self.results_metadata = []
        self.index = None

    def load_search_results(self, question, expansions, sql_filter):
        print("Running slow search...")
        df = finweb_slow_search(
            question=question,
            expansions=expansions,
            sql_filter=sql_filter,
            n_results=1000,
            n_probes=300,
            n_contextify=512,
            algorithm="hybrid-1"
        )
        if df is None:
            raise RuntimeError("No search results.")
        self.search_results_df = df
        self.results_metadata = df.to_dicts()
        print(f"Loaded {len(self.results_metadata)} search results.")

    def build_index(self):
        from usearch.index import Index

        if self.search_results_df is None:
            raise ValueError("No search results DF to build index from.")

        contents = self.search_results_df["content"].to_list()
        batch_size = 16
        all_embs = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i: i + batch_size]
            try:
                e = remotely_embed_sentences_sync(batch)
            except Exception as e:
                print(f"Error embedding batch at index {i}: {e}")
                continue
            all_embs.append(e)
            time.sleep(0.1)

        if not all_embs:
            raise RuntimeError("No embeddings generated!")
        embs = np.vstack(all_embs)
        print(f"Generated embeddings shape: {embs.shape}")  # e.g. (1000, 384)

        # USearch index: specify dimension
        self.index = Index(ndim=embs.shape[1])  # e.g. 384

        for i in range(embs.shape[0]):
            vector = embs[i]
            # ensure float32 and contiguous
            vector = np.asarray(vector, dtype=np.float32)
            vector = np.ascontiguousarray(vector)
            # debug
            #print(f"Vector {i}: shape {vector.shape}, dtype {vector.dtype}, type: {type(vector)}")

            # label first, then vector
            self.index.add(i, vector)

        print("Index built successfully.")

    def get_context(self, query, top_k=5):
        from usearch.index import Index
        if self.index is None:
            raise ValueError("Index not built yet.")
        q = remotely_embed_sentences_sync([query])[0]
        results = self.index.search(q, k=top_k)
        out = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            doc = self.results_metadata[doc_id]
            snippet = doc["content"][:500]
            citation = f"[{rank}] {doc.get('title', 'No Title')} ({doc.get('url', 'No URL')})"
            out.append({"citation": citation, "snippet": snippet})
        return out

    def answer_question(self, query, top_k=5, show_context=False):
        citems = self.get_context(query, top_k=top_k)
        cstr = "\n\n".join([f"{it['citation']}\n{it['snippet']}" for it in citems])
        prompt = (
            "You are an expert research assistant. Use the following sources to answer the question. "
            "Include citations (using the source numbers) in your answer.\n\n"
            f"Question: {query}\n\n"
            f"Sources:\n{cstr}\n\n"
            "Answer:"
        )
        if show_context:
            print("===== PROMPT =====")
            print(prompt)
            print("==================")

        return generate(prompt)

    def interactive_loop(self):
        print("\n=== Entering Interactive Q&A ===")
        while True:
            q = input("Your question (type 'quit' to exit): ").strip()
            if q.lower() in ["quit", "exit"]:
                break
            ans = self.answer_question(q, top_k=5)
            print("\n--- Answer ---")
            print(ans)


def main():
    agent = LongContextAgent()

    question = "What are the terms of the partnership between Microsoft and OpenAI?"
    expansions = [
        "What are the terms and conditions of the deal between Microsoft and OpenAI?",
        "What have Microsoft and OpenAI agreed to in terms of their partnership?",
        "What restrictions have been placed upon OpenAI as part of their Microsoft deal?",
        "Under what conditions can Microsoft or OpenAI exit their partnership?",
        "What are the details of the partnership between OpenAI and Microsoft?",
        "Why did Microsoft decide to invest into OpenAI? What are the terms of the deal?",
        "What are the legal requirements placed upon OpenAI as part of the Microsoft investment?",
        "What are the financial terms and equity stakes in the Microsoft-OpenAI deal?",
        "How does the Microsoft-OpenAI partnership affect OpenAI's autonomy and projects?",
        "What legal requirements must OpenAI meet due to the Microsoft investment?"
    ]
    sql_filter = "SELECT loc FROM engine WHERE published>='2025-01-01'"

    agent.load_search_results(question, expansions, sql_filter)
    agent.build_index()
    agent.interactive_loop()

if __name__ == "__main__":
    main()
