***
# Long Context Answer Agent :robot:

Hi! Welcome to my Long Context Answer Agent.

This project is an  intelligent question-answering agent that leverages the FinWeb semantic search engine to fetch and process a large volume of documents—up to 10,000 per query—to serve as context for answering user questions. 

***
## Index

- [Long Context Answer Agent :robot:](#long-context-answer-agent-robot)
  - [Index](#index)
  - [Functionality](#functionality)
  - [Tools](#tools)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)
***
## Functionality 

1) **User Input:**  The user provides a main question, and a date range (to filter by
publication date).
2) **Slow Search:** The system then uses these inputs to search the Finweb API's /slow-
search/ endpoint. This endpoint returns an encrypted file (stored on S3) along with
an encryption key.
3) **Generate Embeddings:** Downloads and decrypts the file to obtain up to 10,000
search results, then uses the HuggingFace servers to embed the search results.
4) **USearch Index:**  Create a USearch index using the HuggingFace embeddings.
5) **Answer:** When a user asks a question, the USearch index is used to retrieve
relevant search results. The relevant search results, as well as the user's question,
will be sent to a model on OpenRouter to answer the question. The
answer generated includes citations that reference the original search results
fetched from the slow search endpoint.
1) **Multi-turn Conversation:** Allow the user to ask follow-up questions. The follow-up
questions can use the same USearch index.
***
## Tools
- [Finweb API](https://www.nosible.ai/search/v1/docs/swagger) will be used as the search engine to retrieve context for the user’s query.
- [USearch](https://github.com/unum-cloud/usearch), which allows you to build small, single-file semantic search engines.
- [HuggingFace](https://huggingface.co/) servers to generate the embeddings for the USearch index.
- [OpenRouter](https://openrouter.ai/) for LLM model access.
***

## Usage

Before usage, you install dependencies:

```bash
pip install -r requirements.txt
```

You must also create a file called ```keys.env``` that will contain all your api keys as follows, make sure to remember to replace with your own keys:
```env
FINWEB_V1_API_KEY=your_finweb_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
From there all you need to do is run the code, either in your code editor environment, or through the command line:
```bash
python main.py
```
or
```bash
py main.py
```
From there you can use the script as you'd expect, you'll be prompted to put in your question and date range, and any follow up questions in the command line.

If you want a deeper look into the context returned by the FinWeb API, check the ```user_question.csv``` file that is created when you use the script.
***
## Acknowledgements

Big thanks to [Nosible](https://www.nosible.ai/), who set this project for me as a technical interview, I had a lot of fun! :tada: