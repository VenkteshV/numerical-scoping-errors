import json
import time
from typing import List, Dict, Any
from llm.prompt_handler import PromptHandler
from llm.llm_handler import LLMHandler

class FactCheckingHandler:
    """
    Class that manages fact checking task.
    
    Internal States:
    - dataset: "quantemp" or "finqa"
    """
    def __init__(
            self):

        self.prompt_handler = PromptHandler(task="fc")
        self.llm_handler = LLMHandler()
        self.system_prompt = self.prompt_handler.get_system_message()

    def color_text(self, text, color_code):
        """Function used to color cmd line text."""
        return f"\033[{color_code}m{text}\033[0m"
    
    def fact_check(self, claim: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]], on_tok=None) -> str:
        """Runs a single fact-check query."""
        user_prompt = self.prompt_handler.get_user_message(claim, evidence_list, fc_example_list)
        response, total_time = self.llm_handler.generate_response(self.system_prompt, user_prompt, on_token=on_tok)
        return response, total_time
    
    def run(self, input_json_file: str, output_json_file: str, log=True, checkpoint=20) -> List[Dict[str, Any]]:
        """Runs fact-checking on all queries in a input json file"""
        start_time = time.time()
        if log:
            print("Starting fact_checking processing...")
        
        output = {'input_json_file': input_json_file, 'outputs': []}
        with open(input_json_file) as f:
            input_json = json.load(f)

        def on_tok(token):
            print(self.color_text(token, '96'), end="") # colored light cyan

        if not log:
            on_tok = None

        for i, query in enumerate(input_json):
            if log:
                print(self.color_text(f"{i} Fact-checking: {query['claim']}", "93"))

            response, total_time = self.fact_check(query['claim'], query['evidences'], query['examples'], on_tok=on_tok)
            output['outputs'].append({'response': response, 'total_time': total_time})

            if log:
                print(self.color_text(f"\nTotal time: {total_time}\n", "93"))

            if i % checkpoint == 0:
                with open(output_json_file, "w") as f:
                    json.dump(output, f)
                print(self.color_text(f"\nSaved checkpoint!\n", "93"))

        
        total_time = time.time() - start_time
        output['total_time'] = total_time
        with open(output_json_file, "w") as f:
            json.dump(output, f)

        if log:
            print(f"Fact_checking processing finished in {total_time} seconds...")

        return output