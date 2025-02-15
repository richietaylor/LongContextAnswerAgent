import requests
import numpy as np
import json


def remotely_embed_sentences_sync(
        sentences: list,
        endpoint_url: str = None,
        cache_ex: int = 0,
        return_as: str = "numpy"
) -> np.ndarray:
    """
    Given a list of sentences this returns the embeddings as a numpy array.

    :param sentences: the sentences we want to embed.
    :param endpoint_url: the URL of the huggingface endpoint.
    :param cache_ex: the expiration time of caching in seconds.
    :param return_as: what to return the response as.
    :return: the embeddings as a float32 numpy array.
    """
    if len(sentences) > 16:
        # Raise an exception about passing in too many sentences.
        raise ValueError("You may only embed up to 16 sentences at a time.")

    if endpoint_url is None:
        endpoint_url = (
            "https://j3rdbna8u6w2m9zq.us-east-1"
            ".aws.endpoints.huggingface.cloud"
        )

    response = requests.post(
        url=endpoint_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        json={
            "inputs": sentences
        },
        timeout=(5, 15)
    )

    if return_as == "request":
        return response

    # Get the content and load with orjson.
    result = json.loads(response.content)
    return np.array(result, dtype=np.float32)


if __name__ == "__main__":
    embeddings = remotely_embed_sentences_sync(
        sentences=[
            "Alice in Wonderland",
            "The Cheshire Cat"
        ]
    )

    print(embeddings)