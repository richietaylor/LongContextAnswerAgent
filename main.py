import datetime as dt
import os
import time
import requests
import zstandard as zstd # Provides high-speed compression and decompression using the Zstandard algorithm.
import orjson # A fast JSON parser and serializer for handling JSON data efficiently.
import polars as pl # A DataFrame library for efficient data manipulation and analysis (similar to pandas)
from dotenv import load_dotenv # For key handling using .env files
from cryptography.fernet import Fernet
import numpy as np
import json
from usearch.index import Index # Imports the Index class used to create and manage a search index for embedding vectors
import openai
import concurrent.futures
from transformers import AutoTokenizer # Import the Hugging Face tokenizer
import string

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load environment variables from keys.env file
load_dotenv("keys.env")


########################################
# Helper Function for Truncation Using Tokenizer
########################################

def truncate_text(text, max_tokens=255): # max_tokens is set to 255 because the embedding model has a 256 limit
    """
    Truncates the input text using the tokenizer's encode method,
    ensuring the tokenized sequence is no longer than max_tokens.
    Returns the decoded, truncated text.
    """
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
    truncated_text = tokenizer.decode(encoded, skip_special_tokens=True)
    return truncated_text


########################################
# HuggingFace Embedding Functions
########################################

def remotely_embed_sentences_sync(
        sentences: list,
        endpoint_url: str = None,
        cache_ex: int = 0,
        return_as: str = "numpy"
) -> np.ndarray:
    """
    Given a list of sentences, returns the embeddings as a numpy array.
    
    The function expects the endpoint to return either:
      - A list of embeddings, e.g. [[0.1, 0.2, ...], [0.3, 0.4, ...]]
      - A list of dictionaries with an "embedding" key, e.g.
          [{"embedding": [0.1, 0.2, ...]}, {"embedding": [0.3, 0.4, ...]}]
      - A dictionary containing a key "embedding" or "embeddings".
    
    If none of these keys are found and the response is a list of dicts,
    the function will take the value of the first key from each dictionary.
    """
    if len(sentences) > 16:
        raise ValueError("You may only embed up to 16 sentences at a time.")
    
    if endpoint_url is None:
        endpoint_url = (
            "https://j3rdbna8u6w2m9zq.us-east-1.aws.endpoints.huggingface.cloud"
        )
    
    response = requests.post(
        url=endpoint_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        json={"inputs": sentences},
        timeout=(5, 15)
    )
    
    if return_as == "request":
        return response
    
    result = json.loads(response.content)
    
    # If result is a dict, try to extract known keys.
    if isinstance(result, dict):
        if "embedding" in result:
            result = result["embedding"]
        elif "embeddings" in result:
            result = result["embeddings"]
        else:
            result = list(result.values())
    
    # If result is a list of dicts, assume the first key holds the embedding.
    if isinstance(result, list) and result and isinstance(result[0], dict):
        first_key = list(result[0].keys())[0]
        result = [d[first_key] for d in result]
    
    return np.array(result, dtype=np.float32)


def embed_text_hf(texts):
    """
    Embeds a list of texts using the HuggingFace inference endpoint.
    This function first truncates each text to ensure it meets the token limit,
    then batches texts in groups of up to 16 and returns a NumPy array of embeddings.
    The embedding requests are performed in parallel to speed up runtime.
    """
    # Truncate each text using the updated tokenizer-based method (max 265 tokens allowed by endpoint)
    texts = [truncate_text(t, max_tokens=255) for t in texts]
    
    endpoint_url = "https://j3rdbna8u6w2m9zq.us-east-1.aws.endpoints.huggingface.cloud"
    if not endpoint_url:
        raise ValueError("HuggingFace server endpoint not set. Please set HF_EMBED_ENDPOINT in keys.env.")

    batch_size = 16  # Maximum allowed by the endpoint.
    batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]

    # Use ThreadPoolExecutor to run embedding requests in parallel
    # 64 is the sweet-spot for max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(
            lambda batch: remotely_embed_sentences_sync(batch, endpoint_url=endpoint_url, return_as="numpy"),
            batches
        ))
    
    # Concatenate all embedding arrays along the first axis
    return np.concatenate(results, axis=0)


########################################
# Finweb Slow Search Function
########################################

def finweb_slow_search(
    question: str,
    expansions: list,
    sql_filter: str,
    n_results: int = 10_000,
    n_probes: int = 300,
    n_contextify: int = 128,
    algorithm: str = "hybrid-1",
    attempts: int = 0,
) -> pl.DataFrame:
    """
    Makes a slow-search API request to FinWeb V1, then polls until the results are ready.
    Downloads, decrypts, decompresses, and returns the results as a Polars DataFrame.
    """
    finweb_api_key = os.getenv("FINWEB_V1_API_KEY")
    if not finweb_api_key:
        raise ValueError("Finweb API key not set. Please define FINWEB_V1_API_KEY in keys.env.")

    if attempts >= 3:
        print(f"{dt.datetime.utcnow()} The search has failed three times.")
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
            "api-key": finweb_api_key,
        },
    )

    if isinstance(response, requests.Response):
        if response.status_code == 200:
            print(f"{dt.datetime.utcnow()} Slow search was accepted.")
            download_details = response.json()
            download_from: str = download_details["download_from"]
            decrypt_using: str = download_details["decrypt_using"]

            print("")
            print(f"DOWNLOAD LINK: {download_from}")
            print(f"DECRYPTION KEY: {decrypt_using}")
            print("")

            for _ in range(10):
                print(f"{dt.datetime.utcnow()} Checking the download link.")
                dl_response = requests.get(url=download_from)
                if isinstance(dl_response, requests.Response) and dl_response.ok:
                    decompressor = zstd.ZstdDecompressor()
                    fernet = Fernet(decrypt_using.encode("utf-8"))
                    print(f"{dt.datetime.utcnow()} Decrypting the data.")
                    decrypted = fernet.decrypt(dl_response.content)
                    print(f"{dt.datetime.utcnow()} Decompressing the data.")
                    decompressed = decompressor.decompress(decrypted)
                    print(f"{dt.datetime.utcnow()} Deserializing the data.")
                    api_response = orjson.loads(decompressed)
                    print(f"{dt.datetime.utcnow()} Extracting search results.")
                    search_results = []
                    for result in api_response["response"]:
                        semantics = result["semantics"]
                        search_results.append({
                            "question": question,
                            "url_hash": result["url_hash"],
                            "url": result["url"],
                            "title": result["title"],
                            "content": result["content"],
                            "similarity": semantics["similarity"],
                        })
                    return pl.from_dicts(search_results)
                else:
                    print(f"{dt.datetime.utcnow()} Sleeping for another 30s.")
                    time.sleep(30)
            print(f"{dt.datetime.utcnow()} Resubmitting the slow-search.")
            return finweb_slow_search(
                question=question,
                expansions=expansions,
                sql_filter=sql_filter,
                n_results=n_results,
                n_probes=n_probes,
                n_contextify=n_contextify,
                algorithm=algorithm,
                attempts=attempts + 1,
            )
        else:
            handle_api_error(response)
    return None


def handle_api_error(response):
    """Handles errors from Finweb's API responses."""
    if response.status_code == 401:
        print(f"{dt.datetime.utcnow()} API key not authorized:\n{response.text}")
    elif response.status_code == 422:
        print(f"{dt.datetime.utcnow()} Bad API request:\n{response.text}")
    elif response.status_code == 429:
        print(f"{dt.datetime.utcnow()} Rate limit hit:\n{response.text}")
    elif response.status_code in [500, 502, 504]:
        print(f"{dt.datetime.utcnow()} Server error {response.status_code}:\n{response.text}")
    else:
        print(f"{dt.datetime.utcnow()} Unhandled error {response.status_code}:\n{response.text}")


########################################
# Build Expansions via OpenRouter LLM
########################################

def generate_expansions(prompt: str) -> str:
    """
    Generate a response to a prompt using the OpenRouter API.
    
    :param prompt: The prompt to generate a response to.
    :return: The generated response text.
    """
    oai_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    response = oai_client.chat.completions.create(
        model="openai/gpt-4o-2024-11-20",  # this model worked well enough, but it could be switched out
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


def build_expansions(user_question: str) -> list:
    """
    Generates expansions for the user's question by calling the OpenRouter API.
    Returns a list of 5 expansion strings.
    If the output is not in the correct format, it falls back to a default list.
    """
    print(f"Generating expansions for question: {user_question}")

    default_expansions = [ 
        user_question,
        f"What are the details of {user_question}?",
        f"Explain the context of {user_question}",
        f"Why is {user_question} important?",
        f"Information about {user_question}",
    ]
    
    prompt = (
        f"Return a valid JSON array containing exactly 9 distinct, well-phrased expansions for the following question. "
        f"Return only the JSON array with no extra text. The question is: \"{user_question}\"."
    )
    
    try:
        response_text = generate_expansions(prompt)
        # print("DEBUG: Raw expansions response:", response_text)
        cleaned = response_text.strip()
        # Explicitly remove markdown code fences if present:
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        parsed_output = json.loads(cleaned)
        if (isinstance(parsed_output, list) and len(parsed_output) == 9 and 
            all(isinstance(item, str) for item in parsed_output)):
            return [user_question] + parsed_output
        else:
            print("Generated expansions not in expected format. Using default expansions.")
            return default_expansions
    except Exception as e:
        print(f"Error generating expansions via OpenRouter: {e}. Using default expansions.")
        return default_expansions


########################################
# USearch Index and Query Functions
########################################

def build_usearch_index(docs):
    """
    Builds a USearch index using the HuggingFace embeddings.
    `docs` is a list of dictionaries with keys "id" and "content".
    Returns a tuple (index, doc_map) for later retrieval.
    """
    texts = [d["content"] for d in docs]
    doc_ids = [d["id"] for d in docs]
    embeddings = embed_text_hf(texts)  # shape: (N, dim)

    # Use the imported Index class
    index = Index(ndim=len(embeddings[0]))
    # Remove the reserve call if not available.
    doc_map = {}
    for i, (doc_id_str, emb) in enumerate(zip(doc_ids, embeddings)):
        index.add(i + 1, emb)
        doc_map[i + 1] = doc_id_str
    return index, doc_map


def search_usearch_index(query_text, index, doc_map, top_k=5):
    """
    Embeds the query text, then searches the USearch index for the top_k nearest docs.
    Returns a list of tuples (doc_id, distance).
    """
    query_embedding = embed_text_hf([query_text])[0]
    match = index.search(query_embedding, top_k)
    # Access the keys and distances attributes from the Match object.
    keys = match.keys  
    distances = match.distances  
    results = []
    for key, dist in zip(keys, distances):
        str_id = doc_map[key]
        results.append((str_id, dist))
    return results


def show_context(chosen_docs):
    """
    Prints the context (document information) that will be sent to the language model.
    """
    print("\n=== CONTEXT CHUNKS ===")
    for doc in chosen_docs:
        print(f"- [Chunk ID: {doc['id']}]")
        # print(f"  Similarity: {doc['similarity']}")
        print(f"  Title: {doc['title']}")
        print(f"  URL: {doc['url']}")
        print(f"  Content snippet: {doc['content'][:200]}...\n")
    print("======================\n")


def build_prompt(user_question, chosen_docs):
    """
    Builds a prompt that includes the context with citations and the user's question.
    """
    context_text = []
    for doc in chosen_docs:
        label = f"[{doc['id']}]"
        context_text.append(f"{label}: {doc['content']}")
    combined_context = "\n\n".join(context_text)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
        When you use information from a chunk, cite it by referencing its label in square brackets, e.g. [doc3].

        CONTEXT:
        {combined_context}

        QUESTION:
        {user_question}

        Please provide a concise answer with clear citations.
        If the answer isn't contained in the context, say so explicitly.
        """
    return prompt


########################################
# OpenRouter API Call
########################################

def call_openrouter(prompt, openrouter_api_key, model="openai/gpt-4o-2024-11-20", conversation_history=None):
    """
    Calls the OpenRouter API using a chat-completion format.
    If conversation_history is provided, it will be included in the request.
    """
    if conversation_history is None:
        conversation_history = []
    messages = conversation_history + [{"role": "user", "content": prompt}]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openrouter_api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7, # I messed around a bit with temp, around 0.7 seemed to work the best
        "max_tokens": 500,
    }
    try:
        resp = requests.post(url, json=data, headers=headers)
        resp.raise_for_status()
        js = resp.json()
        answer_text = js["choices"][0]["message"]["content"]
        return answer_text
    except Exception as e:
        print("OpenRouter call failed:", e)
        return "Sorry, I had an error calling the model."


########################################
# Main Function
########################################


def main():
    # 1) User Input 

    # Some exmaple questions: 
    # user_question = "What scientific breakthroughs will impact the US markets the most?"
    # user_question = "As a capital seeker offering investors a liquidation preference on their investment, would it be more favourable to you if it was a participating or a non-participating liquidation preference, and why?"
    # user_question = "In investment, what is the difference between a liquidity preference and a liquidation preference?"
    user_question = input("Enter a question to ask the system: ")
    # date_start = "2000-01-01"
    # date_end = "2025-12-31"
    def validate_date(date_text):
        try:
            dt.datetime.strptime(date_text, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    date_start = input("Enter the start date for the search (YYYY-MM-DD): ")
    while not validate_date(date_start):
        print("Incorrect date format, should be YYYY-MM-DD")
        date_start = input("Enter the start date for the search (YYYY-MM-DD): ")

    date_end = input("Enter the end date for the search (YYYY-MM-DD): ")
    while not validate_date(date_end):
        print("Incorrect date format, should be YYYY-MM-DD")
        date_end = input("Enter the end date for the search (YYYY-MM-DD): ")
    # dump_context = True  
    dump_context = input("Show context? (y/n): ").lower() == "y"
    
    # Prepare a filename for saving the search results to CSV
    filename = user_question.translate(str.maketrans('', '', string.punctuation)).replace(" ", "_") + ".csv"

    # 2) Get Finweb Slow Search results (or load from CSV in debug mode)
    if os.path.exists(filename):
        print("This Question has been asked before: Loading documents from CSV.")
        df = pl.read_csv(filename)
    else:
        expansions = build_expansions(user_question)
        print(expansions)
        sql_filter = f"SELECT loc FROM engine WHERE published BETWEEN '{date_start}' AND '{date_end}'"
        df = finweb_slow_search(
            question=user_question,
            expansions=expansions,
            sql_filter=sql_filter,
            n_results=10_000,
            n_probes=300,
            n_contextify=256, # Played around a bit, 256 seemed to work the best
            algorithm="hybrid-1"
        )
        if df is None or df.is_empty():
            print("No results returned from slow search. Exiting.")
            return
        # Filter to the top 5000 by similarity:
        df = df.sort(by="similarity", descending=True).head(5000) #change this for more or less context
        # Save the DataFrame to CSV for if the same question is asked in the future.
        df.write_csv(filename)
        print(f"Saved {len(df)} docs to {filename}.")

    # 3) Build docs list from the DataFrame
    docs = []
    for idx, row in enumerate(df.iter_rows(named=True)):
        doc_id = f"doc{idx}"
        docs.append({
            "id": doc_id,
            "url": row["url"],
            "title": row["title"],
            "content": row["content"]
        })
    print(f"Building USearch index from {len(docs)} docs...")
    index, doc_map = build_usearch_index(docs)
    print("USearch index built successfully.")

    # 4) Answer Queries (Multi-Turn Conversation)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OpenRouter API key not set. Please define OPENROUTER_API_KEY in keys.env.")
    conversation_history = []
    while True:
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        results = search_usearch_index(query_text=user_question, index=index, doc_map=doc_map, top_k=5)
        chosen_docs = []
        for (doc_id_str, dist) in results:
            doc_data = next(d for d in docs if d["id"] == doc_id_str)
            chosen_docs.append(doc_data)
        # print("DEBUG: Chosen docs:", chosen_docs)
        if dump_context:
            show_context(chosen_docs)
        prompt = build_prompt(user_question, chosen_docs)
        # encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True)
        # print(f"Context token count: {len(encoded_prompt)} tokens")

        model_choice = "openai/gpt-4o-2024-11-20" # This model worked well enough, although it could be swapped out if needed
        answer_text = call_openrouter(
            prompt=prompt,
            openrouter_api_key=openrouter_api_key,
            model=model_choice,
            conversation_history=conversation_history
        )
        print("\n=== ANSWER ===")
        print(answer_text)
        print("==============\n")
        conversation_history.append({"role": "user", "content": user_question})
        conversation_history.append({"role": "assistant", "content": answer_text})
        user_question = input("Enter a follow-up question (or 'exit' to quit): ")

if __name__ == "__main__":
    main()
