import os
import sys
import json
import time
from typing import List, Dict, Any
from validation.validation_handler import ValidationHandler
from llm.prompt_handler import PromptHandler
from llm.llm_handler import LLMHandler

class NumericalReasoningHandler:
    """
    Class that manages numerical reasoning tasks.
    
    Internal States:
    - task: "fc" or "qa"
    """
    def __init__(
            self,
            task="fc"):

        self.task = task
        self.validation_handler = ValidationHandler(task=task)
        self.llm_handler = LLMHandler()
        self.prompt_handler = PromptHandler(task=task)
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

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and 'golden' in output_json and output_json['input_json_file'] == input_json_file and output_json['golden'] == golden:
                if log:
                    print("Previous checkpoint loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                sys.exit()
        else:
            output = {'input_json_file': input_json_file, 'golden': golden, 'outputs': []}
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
                    response, total_time = self.process_query(query['claim'], query['qrel_evidences'], query['examples'], on_tok=on_tok)
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
    
    def validate(self, input_json_file: str, output_json_file: str, log: bool=True):
        if log:
            print(f"Starting validation...")

        with open(input_json_file, "r") as f:
            input_json = json.load(f)
        
        with open(output_json_file, "r") as f:
            output_json = json.load(f)

        final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output_json, log=False)

        if self.task == "fc":
            # validate fc
            n_valid, cf_matrix, accuracy, weighted_f1, macro_f1, per_class_f1s, per_taxonomy_f1s = self.validation_handler.validate_classification(input_json, final_output_dicts, log=log)

            output_json['validation'] = {
                'n_valid': n_valid,
                'cf_matrix': cf_matrix,
                'accuracy': accuracy,
                'weighted_f1': weighted_f1,
                'macro_f1': macro_f1,
                'per_class_f1s': per_class_f1s,
                'per_taxonomy_f1s': per_taxonomy_f1s,
                'n_invalid': n_invalid,
                'n_not_started': n_not_started,
                'n_incoherent_separators': n_incoherent_separators,
                'n_incoherent_labels': n_incoherent_labels,
            }
        else:
            # validate qa
            n_valid, em_accuracy, n_invalid_ground_truth = self.validation_handler.validate_exact_match(input_json, final_output_dicts, log=log)

            output_json['validation'] = {
                'n_valid': n_valid,
                'em_accuracy': em_accuracy,
                'n_invalid': n_invalid,
                'n_not_started': n_not_started,
                'n_incoherent_separators': n_incoherent_separators,
                'n_incoherent_labels': n_incoherent_labels,
                'n_invalid_ground_truth': n_invalid_ground_truth,
            }

        try:
            with open(output_json_file, "w") as f:
                json.dump(output_json, f)
        except Exception as e:
            print(f"ERROR with saving validation results, will revert to previous. {e}")
            output_json['validation'] = {}
            with open(output_json_file, "w") as f:
                json.dump(output_json, f)

        if log:
            print(f"Finished validation.")

