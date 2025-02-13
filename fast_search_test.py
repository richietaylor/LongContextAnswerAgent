import requests
import json

url = "https://www.nosible.ai/search/v1/fast-search"
def load_api_keys(path="keys.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data

api_keys = load_api_keys("keys.json")
fast_search_key = api_keys["FINWEB_V1_API_KEY"]

header = {
  "api-key" : fast_search_key,
}
payload = {

  "algorithm": "hybrid-1",
  "algorithm_kwargs": {},
  "expansions": [],
  "n_contextify": 128,
  "n_probes": 10,
  "n_results": 10,
  "question": "What is recent news about South Africa?",
  "sql_filter": "SELECT loc FROM engine"
}

response = requests.post(url, headers=header,json=payload)

if response.status_code == 200:
    print("success")
    output = response.json()["response"]
    for i in range(0,10):
        print((i+1), output[i]["title"])
else:
    print("error!")