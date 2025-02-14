import datetime as dt
import os
import time
from dotenv import load_dotenv
import orjson
import polars as pl
import requests
import simplejson
import zstandard as zstd
from cryptography.fernet import Fernet
from openai import OpenAI
from openai import APIError, OpenAI, RateLimitError
import requests
from json.decoder import JSONDecodeError

load_dotenv(dotenv_path='keys.env')


def finweb_slow_search(
        question: str,
        expansions: list,
        sql_filter: str,
        n_results: int = 10_000,
        n_probes: int = 300,
        n_contextify: int = 128,
        algorithm: str = "hybrid-1",
        attempts: int = 0
) -> pl.DataFrame:
    """
    This method makes a `slow-search` API request to FinWeb V1,
    polls until results are ready, then downloads, decrypts, and decompresses them.
    Returns a polars.DataFrame of the results.
    """
    if attempts >= 3:
        print(f"{dt.datetime.utcnow()} the search has failed three times.")
        return None  # Don't try again, something must be wrong.

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
            "api-key": os.environ["FINWEB_V1_API_KEY"]
        }
    )

    if isinstance(response, requests.Response):

        if response.status_code == 200:

            print(f"{dt.datetime.utcnow()} slow search was accepted.")

            # Get the download details.
            download_details = response.json()
            download_from: str = download_details["download_from"]
            decrypt_using: str = download_details["decrypt_using"]

            print("")
            print(f"DOWNLOAD LINK: {download_from}")
            print(f"DECRYPTION KEY: {decrypt_using}")
            print("")

            for _ in range(10):
                print(f"{dt.datetime.utcnow()} checking the download link.")

                # Fetch the object from the S3 bucket.
                dl_response = requests.get(url=download_from)
                if isinstance(dl_response, requests.Response) and dl_response.ok:

                    # Decrypt and decompress
                    decompressor = zstd.ZstdDecompressor()
                    fernet = Fernet(decrypt_using.encode("utf-8"))

                    print(f"{dt.datetime.utcnow()} decrypting the data.")
                    decrypted = fernet.decrypt(dl_response.content)

                    print(f"{dt.datetime.utcnow()} decompressing the data.")
                    decompressed = decompressor.decompress(decrypted)

                    print(f"{dt.datetime.utcnow()} deserializing the data.")
                    api_response = orjson.loads(decompressed)

                    # Collect search results
                    print(f"{dt.datetime.utcnow()} extracting search results.")
                    search_results = []
                    for result in api_response["response"]:
                        # Get the semantic score dictionary.
                        semantics = result["semantics"]
                        search_results.append(
                            {
                                "question": question,
                                "url_hash": result["url_hash"],
                                "url": result["url"],
                                "title": result["title"],
                                "content": result["content"],
                                "similarity": semantics["similarity"],
                            }
                        )
                    return pl.from_dicts(search_results)

                else:
                    print(f"{dt.datetime.utcnow()} sleeping for another 30s.")
                    time.sleep(30)  # Wait another 30 seconds then check again.

            print(f"{dt.datetime.utcnow()} resubmitting the slow-search.")
            return finweb_slow_search(
                question=question,
                expansions=expansions,
                sql_filter=sql_filter,
                n_results=n_results,
                n_probes=n_probes,
                n_contextify=n_contextify,
                algorithm=algorithm,
                attempts=attempts + 1
            )

        else:
            handle_api_error(response)

    # If we somehow get here and it's not a requests.Response
    return None


def handle_api_error(response):
    """ Simple helper to decode different error statuses from Finweb. """
    if response.status_code == 401:
        print(f"{dt.datetime.utcnow()} YOUR API KEY IS NOT AUTHORIZED \n\n {str(response.text)}")
    elif response.status_code == 422:
        print(f"{dt.datetime.utcnow()} YOU MADE A BAD API REQUEST \n\n {str(response.text)}")
    elif response.status_code == 429:
        print(f"{dt.datetime.utcnow()} YOU HAVE HIT YOUR RATE LIMIT \n\n {str(response.text)}")
    elif response.status_code == 500:
        print(f"{dt.datetime.utcnow()} AN UNEXPECTED ERROR OCCURRED \n\n {str(response.text)}")
    elif response.status_code == 502:
        print(f"{dt.datetime.utcnow()} FINWEB IS CURRENTLY RESTARTING \n\n {str(response.text)}")
    elif response.status_code == 504:
        print(f"{dt.datetime.utcnow()} FINWEB IS CURRENTLY OVERLOADED \n\n {str(response.text)}")
    else:
        print(f"{dt.datetime.utcnow()} UNHANDLED ERROR {response.status_code} \n\n {str(response.text)}")


def show_context(context_chunks):
    """
    Print the chunked context that will be sent to the LLM, so the user can inspect it.
    """
    print("\n=== CONTEXT CHUNKS ===")
    for ch in context_chunks:
        # print(f"- [Chunk ID: {ch['id']}] Similarity={ch['similarity']:.3f}\n{ch['content'][:200]}...\n")
        print(f"- [Chunk ID: {ch['id']}] Title: {ch['title']}\n  URL: {ch['url']}\n  Similarity: {ch['similarity']:.3f}\n  Content snippet: {ch['content'][:200]}...\n")
    print("======================\n")


def build_prompt(user_question, context_chunks):
    """
    Build a single string prompt that includes labeled chunks for citations,
    plus the user question at the end.
    """
    # Label chunks like [doc0], [doc1], etc.
    context_text = []
    for ch in context_chunks:
        label = f"[{ch['id']}]"
        # Each chunk might have partial content or might be truncated if it's huge.
        context_text.append(f"{label}: {ch['content']}")

    # Join everything into a single string
    combined_context = "\n\n".join(context_text)

    prompt = f"""You are a helpful assistant. Answer the user's question using only the context below.
Whenever you use information from a chunk, cite it by referencing its label in square brackets, e.g. [doc3].

CONTEXT:
{combined_context}

QUESTION:
{user_question}

Please provide a concise, well-structured answer with clear citations.
If the answer isn't contained in the context, say so explicitly.
"""
    return prompt


def build_expansions(user_question: str) -> list[str]:
    """
    Given the user's question, build a few expansions / variations
    that might help Finweb's semantic search find more relevant docs.

    This is a naive example that just adds typical rephrasings like 'details of',
    'explanations for', etc. Feel free to tweak or expand as needed.
    """
    expansions = [
        user_question,  # Include the original question verbatim
        f"What are the details of: {user_question}?",
        f"Explain the context of: {user_question}?",
        f"Why is {user_question} important?",
        f"Give me more information about: {user_question}?",
    ]

    # # Get the API key from the environment variable
    # api_key = os.getenv("DEEPSEEK_V1_API_KEY")

    # if not api_key:
    #     raise ValueError("API key not found. Please set DEEPSEEK_V1_API_KEY in your keys.env file.")

    # client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # question = f"Can you please give me 10 expansions of the question: {user_question}?"
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     # Temperature Partner is cool, maybe use it?: https://api-docs.deepseek.com/quick_start/parameter_settings
        
        
    #     messages=[
    #         # Overall behavior of the assistant
    #         {"role": "system", "content": "You are a helpful assistant, give your answer in the format of a python dictionary with no other text"},
    #         # Question
            
    #         {"role": "user", "content": question},
    #     ],
    #     stream=False
    # )

    # print(response.choices[0].message.content)
    # try:
    #     expansions = orjson.loads(response.choices[0].message.content)
    # except orjson.JSONDecodeError as e:
    #     print(f"Error decoding JSON: {e}")
    #     expansions = []

    # print(response.choices[0].message.content)
    return expansions


# def deepseek_ask(client, prompt, conversation_history=None, timeout=60, max_retries=10, backoff=5):
#     """
#     Make a direct POST to DeepSeek's /v1/chat/completions endpoint
#     with a custom timeout and retry logic.
#     """
#     # Build messages
#     messages = []
#     system_message = {
#         "role": "system",
#         "content": "You are a helpful assistant that cites relevant doc IDs in square brackets."
#     }
#     messages.append(system_message)

#     if conversation_history:
#         messages.extend(conversation_history)

#     messages.append({"role": "user", "content": prompt})

#     # Construct the JSON payload
#     data = {
#         "model": "deepseek-chat",
#         "messages": messages,
#         "stream": False
#     }

#     # Use the client's API key and base_url
#     endpoint = f"{str(client.base_url).rstrip('/')}/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {client.api_key}",
#         "Content-Type": "application/json"
#     }

#     # Retry loop
#     for attempt in range(1, max_retries + 1):
#         try:
#             print(f"[DEBUG] Attempt {attempt}: POST {endpoint} with timeout={timeout}")
#             resp = requests.post(endpoint, json=data, headers=headers, timeout=timeout)
#             print("[DEBUG] Status code:", resp.status_code)
#             print("[DEBUG] Body:", resp.text)

#             # If you got a 2xx status but the body is empty, treat that as an error to maybe retry
#             if resp.status_code == 200 and not resp.text.strip():
#                 print("[DEBUG] 200 with empty body, consider this an error to retry or fail.")
#                 # If you want to treat it as a retryable error:
#                 if attempt < max_retries:
#                     time.sleep(backoff)
#                     continue
#                 else:
#                     raise ValueError("DeepSeek returned 200 with empty body.")

#             resp.raise_for_status()  # raise for 4xx or 5xx

#             # Attempt to parse the JSON
#             parsed = resp.json()
#             # Validate structure
#             if "choices" not in parsed or not parsed["choices"]:
#                 raise ValueError("No 'choices' in DeepSeek response.")
#             return parsed["choices"][0]["message"]["content"]

#         except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
#             print(f"[DEBUG] Network error on attempt {attempt}: {e}")
#             # Retry if attempts remain
#             if attempt < max_retries:
#                 time.sleep(backoff)
#             else:
#                 raise

#         except Exception as e:
#             print(f"[DEBUG] Non-network error on attempt {attempt}: {e}")
#             # Decide if you want to treat it as retryable
#             raise  # or handle differently

#     raise RuntimeError("Ran out of retries, request failed each time.")

import time


def deepseek_ask(client, prompt, conversation_history=None, max_retries=10, backoff=5):
    """
    Call the DeepSeek (OpenAI-like) chat completion API using the official OpenAI client method.
    Includes built-in retry logic and debug prints.
    """
    messages = []
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that cites relevant doc IDs in square brackets."
    }
    messages.append(system_message)

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": prompt})

    # print("[DEBUG] Sending request with messages:")
    # print(messages)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DEBUG] Attempt {attempt}: Calling client.chat.completions.create()")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            print("[DEBUG] Response received:")
            # print(response)

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Response did not contain any choices or content.")

            return response.choices[0].message.content


        except (APIError, RateLimitError, JSONDecodeError) as e:
            # Try to retrieve an HTTP status code from the exception, if available.
            http_status = getattr(e, "http_status", "Unknown")
            print(f"[DEBUG] Error on attempt {attempt} (HTTP status: {http_status}): {e}")
            if attempt < max_retries:
                print(f"[DEBUG] Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                print("[DEBUG] Max retries reached. Raising exception.")
                raise

        except Exception as e:
            print(f"[DEBUG] Non-retryable error on attempt {attempt}: {e}")
            raise
        # except (APIError, RateLimitError) as e:
        #     print(f"[DEBUG] Transient error on attempt {attempt}: {e}")
        #     if attempt < max_retries:
        #         print(f"[DEBUG] Waiting {backoff} seconds before retry...")
        #         time.sleep(backoff)
        #     else:
        #         print("[DEBUG] Max retries reached. Raising exception.")
        #         raise
        # except Exception as e:
        #     print(f"[DEBUG] Non-retryable error on attempt {attempt}: {e}")
        #     raise

    raise RuntimeError("All retry attempts failed.")


def main():
    # ----------------------------------------------------------------------------------
    # 1) Initialize Finweb / DeepSeek API Keys
    # ----------------------------------------------------------------------------------
    finweb_api_key = os.getenv("FINWEB_V1_API_KEY")
    if not finweb_api_key:
        raise ValueError("API key not found. Please set FINWEB_V1_API_KEY in your .env file.")

    deepseek_api_key = os.getenv("DEEPSEEK_V1_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DeepSeek API key not found. Please set DEEPSEEK_V1_API_KEY in your keys.env file.")

    # Create the DeepSeek client
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    # ----------------------------------------------------------------------------------
    # 2) Gather user input for the initial question
    # ----------------------------------------------------------------------------------
    # user_question = input("\nEnter your main question: ")
    user_question = "What scientific breakthroughs will impact the US markets the most?"

    print("Question:", {user_question})
    # ----------------------------------------------------------------------------------
    # 3) Run Finweb slow search (only once for the main question)
    # ----------------------------------------------------------------------------------
    expansions = build_expansions(user_question)

    print("Expansions:", expansions)

    # expansions = [
    #     "What are the terms and conditions of the deal between Microsoft and OpenAI?",
    #     "What have Microsoft and OpenAI agreed to in terms of their partnership?",
    #     "What restrictions have been placed upon OpenAI as part of their Microsoft deal?",
    #     "Under what conditions can Microsoft or OpenAI exit their partnership?",
    #     "What are the details of the partnership between OpenAI and Microsoft?",
    #     "Why did Microsoft decide to invest into OpenAI? What are the terms of the deal?",
    #     "What are the legal requirements placed upon OpenAI as part of the Microsoft investment?",
    #     "What are the financial terms and equity stakes in the Microsoft-OpenAI deal?",
    #     "How does the Microsoft-OpenAI partnership affect OpenAI's autonomy and projects?",
    #     "What legal requirements must OpenAI meet due to the Microsoft investment?"
    # ]

    # Ask the user for a date range
    # date_start = input("Enter the start date (YYYY-MM-DD): ")
    date_start = "2025-01-01"
    # date_end = input("Enter the end date (YYYY-MM-DD): ")
    date_end = dt.datetime.now().strftime("%Y-%m-%d")
    print("Date range:", date_start, "to", date_end)

    sql_filter = f"SELECT loc FROM engine WHERE published BETWEEN '{date_start}' AND '{date_end}'"
    

    # sql_filter = "SELECT loc FROM engine WHERE published>='2025-01-01' ORDER BY similarity DESC"
    sql_filter = "SELECT loc FROM engine WHERE published>='2025-01-01'"


    df = finweb_slow_search(
        question=user_question,
        expansions=expansions,
        sql_filter=sql_filter,
        n_results=10_000,      # can be up to 10,000
        n_probes=300,
        n_contextify=512,
        algorithm="hybrid-1"
    )

    if df is None or df.is_empty():
        print("No results returned from slow search. Exiting.")
        return

    # ----------------------------------------------------------------------------------
    # 4) Sort by similarity and pick top results to reduce hallucinations
    # ----------------------------------------------------------------------------------
    df = df.sort(by="similarity", descending=True)
    # Grab top 50 or so; adjust as needed
    top_df = df.head(50)

    # ----------------------------------------------------------------------------------
    # 5) Prepare context chunks (one chunk = one row for simplicity)
    # ----------------------------------------------------------------------------------
    context_chunks = []
    for idx, row in enumerate(top_df.iter_rows(named=True)):
        chunk_id = f"doc{idx}"
        context_chunks.append({
            "id": chunk_id,
            "content": row["content"],
            "similarity": row["similarity"],
            "url": row["url"],
            "title": row["title"]
        })

    # ----------------------------------------------------------------------------------
    # 6) Let the user optionally inspect context
    # ----------------------------------------------------------------------------------
    # choice = input("Show the context that will be sent to DeepSeek? (y/n) ")
    choice = "y"
    if choice.lower().startswith("y"):
        show_context(context_chunks)

    # ----------------------------------------------------------------------------------
    # 7) Build prompt and call DeepSeek for the first answer
    # ----------------------------------------------------------------------------------
    prompt = build_prompt(user_question, context_chunks)
    answer = deepseek_ask(client, prompt)
    print("\n=== ANSWER ===")
    print(answer)
    print("=============\n")

    # ----------------------------------------------------------------------------------
    # 8) Multi-turn follow-up loop
    # ----------------------------------------------------------------------------------
    conversation_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]

    while True:
        follow_up = input("Enter a follow-up question (or 'exit' to quit): ")
        if follow_up.strip().lower() == "exit":
            print("Exiting. Goodbye!")
            break

        # We can reuse the same context chunks or re-sort them based on new question.
        # For simplicity, let's just reuse the same top_df.
        follow_up_prompt = build_prompt(follow_up, context_chunks)

        # Optionally show context again if user wants to see it:
        # show_context(context_chunks)

        follow_up_answer = deepseek_ask(
            client=client,
            prompt=follow_up_prompt,
            conversation_history=conversation_history
        )

        # Print the result
        print("\n=== ANSWER ===")
        print(follow_up_answer)
        print("=============\n")

        # Update conversation history
        conversation_history.append({"role": "user", "content": follow_up_prompt})
        conversation_history.append({"role": "assistant", "content": follow_up_answer})


if __name__ == "__main__":
    # Just call our main driver function
    main()
