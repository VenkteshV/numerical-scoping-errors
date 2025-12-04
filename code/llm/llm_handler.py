import json
import time
import requests
from typing import Callable, Dict, Any

class LLMHandler:
    """
    Class that handles connections to an LLM through LMStudio.
    
    Internal States:
    - completion_params: json of parameters for the LLM
    - log_stats: whether to log statistics of operation or not.
    """
    def __init__(
            self,
            completion_params_file: str = "llm/completion_params.json",
            log_stats: bool = False):

        self.completion_params = self.load_completion_params(completion_params_file)
        self.log_stats = log_stats
        
        # LMStudio typically runs on localhost:1234
        self.api_url = "http://localhost:1234/v1/chat/completions"
        self.session = requests.Session()

    def load_completion_params(self, file_path: str) -> Dict[str, Any]:
        """Loads the parameter json file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def generate_response(
            self,
            system_prompt: str,
            user_prompt: str,
            on_token: Callable[[str], None] = None) -> str:
        """Queries LMStudio LLM to get AI response."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        payload = {
            "model": self.completion_params["model"],
            "messages": messages,
            "stream": True,
            **self.completion_params.get("parameters", {})
        }

        self.payload = json.dumps(payload, indent=4)

        start_time = time.time()
        collected_messages = []

        try:
            # NOTE: leave read timeout None for long streams. Connect timeout small.
            response = self.session.post(self.api_url, json=payload, stream=True, timeout=(3.05, None))

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data['choices'][0]['finish_reason'] is not None:
                            break
                        token = data['choices'][0]['delta'].get('content', '')
                        if token:
                            collected_messages.append(token)
                            if on_token:
                                on_token(token)
                        
                            if self.log_stats:
                                chunk_time = time.time() - start_time
                                print(f"Token received {chunk_time:.2f} seconds after request: {token}")

            total_time = time.time() - start_time

            full_response = ''.join(collected_messages)

            if self.log_stats:
                print(f"Full response received {total_time:.2f} seconds after request")
                print(f"Full response: {full_response}")

            return full_response, total_time
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"ERROR: {e}", time.time() - start_time