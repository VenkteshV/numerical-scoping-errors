import json
import time
import threading
from typing import Callable, Dict, Any, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StopOnStrings(StoppingCriteria):
    """
    Stops generation once ANY of the stop strings appears in the decoded output.
    This checks the *full* decoded text so far (simple + reliable, slightly inefficient).
    """

    def __init__(self, tokenizer, stop_strings: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = [s for s in stop_strings if s]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_strings:
            return False
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


class LLMHandler:
    """
    Class that handles local GPU inference using Hugging Face Transformers.

    Internal States:
    - completion_params: json of parameters for the LLM
    - log_stats: whether to log statistics of operation or not.
    """

    def __init__(
        self,
        completion_params_file: str = "llm/completion_params.json",
        log_stats: bool = False,
        device: Optional[str] = None,
    ):
        self.completion_params = self.load_completion_params(completion_params_file)
        self.log_stats = log_stats
        self.context_length = 8192

        # Choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Model id should be a Hugging Face repo id or a local path.
        # Example: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" or "deepseek-r1-distill-qwen-7b" for LMStudio
        self.model_id = self.completion_params["model"]

        # Load tokenizer/model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, trust_remote_code=True)

        # If the tokenizer has no pad token, set it to eos to keep generate() happy.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use device_map="auto" for multi-GPU nodes; otherwise it will place on one GPU.
        # torch_dtype="auto" picks bf16/fp16 where appropriate on modern GPUs.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        self.model.eval()

        # If we're on CPU or device_map wasn't used, move the model explicitly.
        if self.device != "cuda":
            self.model.to(self.device)

    def load_completion_params(self, file_path: str) -> Dict[str, Any]:
        """Loads the parameter json file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Uses the model's chat template if available; otherwise falls back to a simple format.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback (works but may reduce quality for chat-tuned models)
        system = messages[0]["content"]
        user = messages[1]["content"]
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] = None,
    ) -> Tuple[str, float]:
        """Runs local HF model generation with streaming tokens."""

        params = dict(self.completion_params.get("parameters", {}))
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 0.95))
        requested_max_new_tokens = int(params.get("max_tokens", 256))
        repetition_penalty = float(params.get("repeat_penalty", 1.0))
        stop_strings = params.get("stop", []) or []

        # These are OpenAI-style params; transformers doesn't support them directly in the same way.
        # Keeping them in payload for transparency.
        frequency_penalty = params.get("frequency_penalty", 0)
        presence_penalty = params.get("presence_penalty", 0)

        messages = self._build_messages(system_prompt, user_prompt)
        prompt_text = self._format_prompt(messages)

        # Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.context_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs['input_ids'].shape[-1]
        max_new_tokens = min(requested_max_new_tokens, max(self.context_length - prompt_len, 0))

        if max_new_tokens == 0:
            return "ERROR: 'choices'", 0.0

        # Stopping criteria for stop strings (optional)
        stopping = StoppingCriteriaList()
        if stop_strings:
            stopping.append(StopOnStrings(self.tokenizer, stop_strings))

        # Streamer yields text chunks as they are generated
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation config (kept close to your payload concept)
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            stopping_criteria=stopping if len(stopping) > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Save a debug "payload" similar to your LMStudio json
        self.payload = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "stream": True,
                "context_length": self.context_length,
                "max_new_tokens": max_new_tokens,
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": repetition_penalty,
                    "stop": stop_strings,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                },
            },
            indent=4,
        )

        start_time = time.time()
        collected: List[str] = []

        try:
            # Run generation in a background thread so we can iterate the streamer in this thread.
            t = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            t.start()

            for chunk in streamer:
                if not chunk:
                    continue

                collected.append(chunk)
                if on_token:
                    on_token(chunk)

                if self.log_stats:
                    chunk_time = time.time() - start_time
                    print(f"Token chunk received {chunk_time:.2f} seconds after request: {chunk}")

            t.join()

            total_time = time.time() - start_time
            full_response = "".join(collected)

            # If stop strings are used, trim at the first occurrence for a cleaner return
            if stop_strings:
                earliest = None
                for s in stop_strings:
                    idx = full_response.find(s)
                    if idx != -1:
                        earliest = idx if earliest is None else min(earliest, idx)
                if earliest is not None:
                    full_response = full_response[:earliest]

            if self.log_stats:
                print(f"Full response received {total_time:.2f} seconds after request")
                print(f"Full response: {full_response}")

            return full_response, total_time

        except Exception as e:
            return f"ERROR: {e}", time.time() - start_time
