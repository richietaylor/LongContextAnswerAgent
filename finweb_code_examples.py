# ======================================================================================================================
# THIS FINWEB EXAMPLE IS FOR TAKE HOMES. IT DEMONSTRATES HOW SLOW SEARCHES CAN BE RUN BACK-TO-BACK. AFTER EACH REQUEST
# IS MADE THE DOWNLOAD LINK IS POLLED FOR UP TO 5 MINUTES BEFORE THE REQUEST IS RESUBMITTED. WHEN THE REQUEST COMPLETES
# THE SEARCH RESULTS ARE DOWNLOADED FROM WASABI S3. THE SEARCH RESULTS ARE ENCRYPTED USING FERNET AND COMPRESSED USING
# Z-STANDARD. THE RESULTS ARE CONCATENATED USING POLARS AND WRITTEN TO DISK AS A CSV FILE FOR DEMONSTRATION PURPOSES.
# IT ALSO DEMONSTRATES HOW TO RUN A FAST SEARCH.
#
# REQUIREMENTS:
#   - orjson==3.9.15
#   - polars==1.1.0
#   - zstandard==0.23.0
#   - cryptography>=42.0.5
#   - simplejson>=3.17.6
# ======================================================================================================================

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

load_dotenv(dotenv_path='keys.env')

def finweb_fast_search(
        question: str,
        expansions: list,
        sql_filter: str,
        n_results: int = 100,
        n_probes: int = 30,
        n_contextify: int = 128,
        algorithm: str = "hybrid-1"
) -> None:
    """
    This method makes a `fast-search` API request to FinWeb V1.

    :param question: the question that we want to search for.
    :param expansions: lexically and semantically similar questions.
    :param sql_filter: the SQL filter to apply for the search.
    :param n_results: the number of results we would like to return. Max is 100.
    :param n_probes: the number of shards we would like to search over. Max 30 for fast search 10 is considered the min.
    :param n_contextify: the amount of text we want to return.  128 - 512 are usually good params.
    :param algorithm: the search algorithm we would like to use.
    :return: a polars DataFrame containing the search results.
    """
    response = requests.post(
        url="https://www.nosible.ai/search/v1/fast-search",
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

            # Convert the response to JSON.
            response_data = response.json()

            # Print out the debug and message.
            print("=" * 80)
            print(question)
            print(response_data["message"])
            print(response.elapsed)
            print("=" * 80)

            for result in response_data["response"]:

                print(simplejson.dumps(result, indent=4))

        elif response.status_code == 401:
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
    This method makes a `slow-search` API request to FinWeb V1. The request will return an S3 download link and a
    decryption key. When the request completes the file will be uploaded to S3 and the link will be made public.
    When that happens this script will see the file, download it, decrypt it, decompress it, and unpack it.

    :param question: the question that we want to search for.
    :param expansions: lexically and semantically similar questions.
    :param sql_filter: the SQL filter to apply for the search.
    :param n_results: the number of results we would like to return. Max is 10_000.
    :param n_probes: the number of shards we would like to search over. Max is 300.
    :param n_contextify: the amount of text we want to return. Good params are similar to fast search.
    :param algorithm: the search algorithm we would like to use.
    :param attempts: the number of attempts that have occurred so far.
    :return: a polars DataFrame containing the search results.
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

            # Get the URL and the decryption key.
            download_from: str = download_details["download_from"]
            decrypt_using: str = download_details["decrypt_using"]

            print("")
            print(f"DOWNLOAD LINK: {download_from}")
            print(f"DECRYPTION KEY: {decrypt_using}")
            print("")

            for i in range(10):

                print(f"{dt.datetime.utcnow()} checking the download link.")

                # Fetch the object from the S3 bucket.
                response = requests.get(url=download_from)

                search_results = []

                if isinstance(response, requests.Response) and response.ok:

                    # Create a decrypter and decompressor.
                    decompressor = zstd.ZstdDecompressor()
                    fernet = Fernet(decrypt_using.encode("utf-8"))

                    # Decrypt the data using the Fernet key.
                    print(f"{dt.datetime.utcnow()} decrypting the data.")
                    decrypted = fernet.decrypt(response.content)

                    # Decompress the data using z-standard decompressor.
                    print(f"{dt.datetime.utcnow()} decompressing the data.")
                    decompressed = decompressor.decompress(decrypted)

                    # Deserialize the data using orjson library.
                    print(f"{dt.datetime.utcnow()} deserializing the data")
                    api_response = orjson.loads(decompressed)

                    # Extract the results into the output DataFrame.
                    print(f"{dt.datetime.utcnow()} extracting search results")

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

                    # Write all the results to a CSV file.
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

        elif response.status_code == 401:
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


if __name__ == "__main__":

    api_key = os.getenv("FINWEB_V1_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set FINWEB_V1_API_KEY in your .env file.")

    os.environ["FINWEB_V1_API_KEY"] = api_key

    df = finweb_slow_search(
        question="What are the terms of the partnership between Microsoft and OpenAI?",
        expansions=[
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
        ],
        sql_filter="SELECT loc FROM engine WHERE published>='2025-01-01'",

        # These default parameters should work fine for your use case.
        n_probes=300,
        n_results=10_000,
        n_contextify=512,
        algorithm="hybrid-1"
    )
    df.write_csv(file="all_results.csv")  # Write all results to disk.

    # You can also use fast search for testing.
    finweb_fast_search(
        question="What is the role of Fixed Income in Portfolio Diversification?",
        expansions=[],
        sql_filter="SELECT loc FROM engine",
        n_probes=10,
        n_results=10,
        n_contextify=128,
        algorithm="hybrid-1"
    )
