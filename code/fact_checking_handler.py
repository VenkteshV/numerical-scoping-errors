import os
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
            self,
            task="fc"):

        self.task = task
        self.prompt_handler = PromptHandler(task=task)
        self.llm_handler = LLMHandler()
        self.system_prompt = self.prompt_handler.get_system_message()

    def color_text(self, text: str, color_code: str) -> str:
        """Function used to color cmd line text."""
        return f"\033[{color_code}m{text}\033[0m"
    
    def process_query(self, query: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]], on_tok=None) -> str:
        """Runs a single fact-check query."""
        user_prompt = self.prompt_handler.get_user_message(query, evidence_list, fc_example_list)
        response, total_time = self.llm_handler.generate_response(self.system_prompt, user_prompt, on_token=on_tok)
        return response, total_time
    
    def run(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=20, golden: bool=False) -> List[Dict[str, Any]]:
        """Runs on all queries in a input json file"""
        if log:
            start_time = time.time()
            print(f"Starting {self.task} processing...")

        # Load previous checkpoint
        if os.path.isfile(output_json_file):
            with open(output_json_file, "r") as f:
                output_json = json.load(f)

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and output_json['input_json_file'] == input_json_file:
                if log:
                    print("Previous checkpoint loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                return None
        else:
            output = {'input_json_file': input_json_file, 'outputs': []}
        start_index = len(output['outputs'])
        if log:
            print(f"Starting from claim with index: {start_index}.")
            
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        # Set up intermediate writing if user wants logs
        def on_tok(token):
            print(self.color_text(token, '96'), end="") # colored light cyan

        if not log:
            on_tok = None

        # Main inference loop
        for i, query in enumerate(input_json[start_index:]):
            if self.task == "fc":
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['claim']}", "93"))

                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query(query['claim'], [query['doc']], query['examples'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query(query['claim'], query['evidences'], query['examples'], on_tok=on_tok)
            else:
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['question']}", "93"))
                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query(query['question'], query['doc'], query['examples'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query(query['question'], query['evidences'], query['examples'], on_tok=on_tok)
            
            output['outputs'].append({'response': response, 'total_time': total_time})

            if log:
                print(self.color_text(f"\nTotal time: {total_time}\n", "93"))

            if i % checkpoint == 0:
                with open(output_json_file, "w") as f:
                    json.dump(output, f)
                print(self.color_text(f"\nSaved checkpoint!\n", "93"))

        with open(output_json_file, "w") as f:
            json.dump(output, f)

        if log:
            total_time = time.time() - start_time
            print(f"Processing finished in {total_time} seconds...")

        return output