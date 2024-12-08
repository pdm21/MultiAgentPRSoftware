import requests
import sseclient
import logging
import json
from typing import Generator, Union, Dict, Any

from dotenv import load_dotenv
import os

load_dotenv() 
api_key = os.getenv('PERPLEXITY_API_KEY')

# ----------------------------------------------------------------
def make_perplexity_api_call(
    api_key: str, 
    model: str, 
    user_message: str, 
    stream: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Union[Dict[str, Any], Generator[str, None, None]]:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    try:
        if stream:
            return _stream_response(url, payload, headers)
        else:
            return _normal_response(url, payload, headers)
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {"error": str(e)}

def _stream_response(url: str, payload: dict, headers: dict) -> Generator[str, None, None]:
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        # Parse JSON line if possible
                        data = json.loads(line)
                        if 'choices' in data and data['choices'][0]['delta'].get('content', ''):
                            yield data['choices'][0]['delta']['content']
                    except json.JSONDecodeError:
                        logging.error(f"Failed to decode JSON: {line}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Stream request failed: {e}")
        yield f"Error: {str(e)}"


def _normal_response(
    url: str, 
    payload: dict, 
    headers: dict
) -> Dict[str, Any]:
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Normal request failed: {e}")
        return {"error": f"Request error: {str(e)}"}


# Streaming Response
streaming_response = make_perplexity_api_call(
    api_key, 
    "llama-3.1-sonar-small-128k-online", 
    "What is the latest news on the Indiana Pacers?", 
    stream=True
)
for chunk in streaming_response:
    print(chunk, end='', flush=True)

# Normal Response
normal_response = make_perplexity_api_call(
    api_key, 
    "llama-3.1-sonar-small-128k-online", 
    "Who is winning at the Olympics?",
    stream=False
)
print(normal_response['choices'][0]['message']['content'])