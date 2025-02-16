import openai
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='keys.env')

ROUTER_API_KEY = os.environ["ROUTER_API_KEY"]


def generate(prompt: str) -> str:
    """
    Generate a response to a prompt using the OpenRouter API.

    :param prompt: The prompt to generate a response to.
    :return: The generated response.
    """
    # Establish the OpenAI client.
    oai_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=ROUTER_API_KEY,
    )

    # Generate the response using the OpenAI client.
    response = oai_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    # Return the response
    return response.choices[0].message.content


if __name__ == "__main__":
    print(generate("What do you know about the universe?"))

