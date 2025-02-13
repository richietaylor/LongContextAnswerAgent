# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# # Load environment variables from keys.env file
# load_dotenv(dotenv_path='keys.env')

# # Get the API key from the environment variable
# api_key = os.getenv("DEEPSEEK_V1_API_KEY")

# if not api_key:
#     raise ValueError("API key not found. Please set DEEPSEEK_V1_API_KEY in your keys.env file.")


# client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",

#     # Temperature Partner is cool, maybe use it?: https://api-docs.deepseek.com/quick_start/parameter_settings

#     messages=[
#         # Overall behavior of the assistant
#         {"role": "system", "content": "You are a helpful assistant"},
#         # Question
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from keys.env file
load_dotenv(dotenv_path='keys.env')

# Get the API key from the environment variable
api_key = os.getenv("DEEPSEEK_V1_API_KEY")

if not api_key:
    raise ValueError("API key not found. Please set DEEPSEEK_V1_API_KEY in your keys.env file.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    # Temperature Partner is cool, maybe use it?: https://api-docs.deepseek.com/quick_start/parameter_settings
    messages=[
        # Overall behavior of the assistant
        {"role": "system", "content": "You are a helpful assistant"},
        # Question
        {"role": "user", "content": "What is happening between openAI and microsoft?"},
    ],
    stream=False
)

print(response.choices[0].message.content)
