import os
import sys
import json
import time
from typing import List, Dict, Any
from validation.validation_handler import ValidationHandler
from llm.prompt_handler import PromptHandler
from llm.llm_handler import LLMHandler
from verifier.verifier_handler import VerifierHandler
from verifier.multitask_verifier_handler import MultiTaskVerifierHandler

class NumericalReasoningHandler:
    """
    Class that manages numerical reasoning tasks.
    
    Internal States:
    - task: "fc" or "qa"
    """
    def __init__(
            self,
            task="fc",
            platform="local"):

        self.task = task
        self.platform = platform

        if platform == "local":
            from llm.llm_handler import LLMHandler
            self.llm_handler = LLMHandler()
        else:
            from llm.llm_handler_delftblue import LLMHandler
            self.llm_handler = LLMHandler()

        self.validation_handler = ValidationHandler(task=task)
        self.prompt_handler = PromptHandler(task=task)
        self.system_prompt = self.prompt_handler.get_system_message()
        self.judge_system_prompt = self.prompt_handler.get_system_message(variant="judge")
        self.judge_new_system_prompt = self.prompt_handler.get_system_message(variant="judge_new")

    def color_text(self, text: str, color_code: str) -> str:
        """Function used to color cmd line text."""
        return f"\033[{color_code}m{text}\033[0m"
    
    def process_query(self, query: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]], on_tok=None) -> str:
        """Runs a single fact-check/qa query."""
        user_prompt = self.prompt_handler.get_user_message(query, evidence_list, fc_example_list)
        response, total_time = self.llm_handler.generate_response(self.system_prompt, user_prompt, on_token=on_tok)
        return response, total_time
    
    def process_query_judge(self, query: str, evidence_list: List[str], response: str, actual_label: str, on_tok=None) -> str:
        """Runs a single fact-check/qa LLM-as-a-judge query."""
        user_prompt = self.prompt_handler.get_user_message(query, evidence_list, fc_example_list=None, metadata={'response': response, 'actual_label': actual_label}, variant="judge")
        response, total_time = self.llm_handler.generate_response(self.judge_system_prompt, user_prompt, on_token=on_tok)
        return response, total_time
    
    def process_query_judge_new(self, query: str, evidence_list: List[str], response: str, actual_label: str, on_tok=None) -> str:
        """Runs a single fact-check/qa LLM-as-a-judge query."""
        user_prompt = self.prompt_handler.get_user_message(query, evidence_list, fc_example_list=None, metadata={'response': response, 'actual_label': actual_label}, variant="judge_new")
        response, total_time = self.llm_handler.generate_response(self.judge_new_system_prompt, user_prompt, on_token=on_tok)
        return response, total_time
    
    def run(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=20, golden: bool=False, evidence_split: str="doc", completion_params: list=None) -> List[Dict[str, Any]]:
        """
        Main function for inference processing

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
            checkpoint (int): how often to save to the output file
            golden (bool): whether to use golden evidence
            evidence_split (str): which evidence split to use
        """
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
            output = {'input_json_file': input_json_file, 'golden': golden, 'outputs': [], 'evidence_split': evidence_split}

        # Check evidence split
        if 'evidence_split' not in output:
            output['evidence_split'] = evidence_split
        else:
            if output['evidence_split'] != evidence_split:
                print(f"ERROR! Output file exists and has evidence_split not matching desired evidence split {evidence_split} {output['evidence_split']}!")
                sys.exit()

        # Load correct completion params if given
        if completion_params:
            previous_completion_params = self.llm_handler.completion_params
            self.llm_handler = LLMHandler(completion_params_file=completion_params)
            output['completion_params'] = completion_params

        # Start processing
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
                    response, total_time = self.process_query(query['claim'], [query[evidence_split]], query['examples'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query(query['claim'], query[evidence_split], query['examples'], on_tok=on_tok)
            else:
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['question']}", "93"))
                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query(query['question'], query[evidence_split], query['examples'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query(query['question'], query[evidence_split], query['examples'], on_tok=on_tok)
            
            output['outputs'].append({'response': response, 'total_time': total_time})

            if log:
                print(self.color_text(f"\nTotal time: {total_time}\n", "93"))

            if i % checkpoint == 0:
                with open(output_json_file, "w") as f:
                    json.dump(output, f)
                print(self.color_text(f"\nSaved checkpoint!\n", "93"))

        with open(output_json_file, "w") as f:
            json.dump(output, f)

        if completion_params:
            self.llm_handler.completion_params = previous_completion_params

        if log:
            total_time = time.time() - start_time
            print(f"Processing finished in {total_time} seconds...")

        return output
    
    def validate(self, input_json_file: str, output_json_file: str, log: bool=True):
        """
        Main function for validation processing.

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
        """
        if log:
            print(f"Starting validation...")

        with open(input_json_file, "r") as f:
            input_json = json.load(f)
        
        with open(output_json_file, "r") as f:
            output_json = json.load(f)

        final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output_json, log=False)

        if self.task == "fc":
            # validate fc
            n_valid, cf_matrix, accuracy, weighted_f1, macro_f1, per_class_f1s, per_taxonomy_f1s, final_output_dicts = self.validation_handler.validate_classification(input_json, final_output_dicts, log=log)

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
            n_valid, em_accuracy, n_invalid_ground_truth, final_output_dicts = self.validation_handler.validate_exact_match(input_json, final_output_dicts, log=log)

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

    def judge(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=20, golden: bool=False, evidence_split: str="doc"):
        """
        Main function for LLM-as-a-judge processing

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
            checkpoint (int): how often to save to the output file
            golden (bool): whether to use golden evidence
            evidence_split (str): which evidence split to use
        """
        if log:
            start_time = time.time()
            print(f"Starting {self.task} JUDGE processing...")

        # Load outputs and previous checkpoint
        if os.path.isfile(output_json_file):
            with open(output_json_file, "r") as f:
                output_json = json.load(f)

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and 'golden' in output_json and 'evidence_split' in output_json and output_json['input_json_file'] == input_json_file and output_json['golden'] == golden and output_json['evidence_split'] == evidence_split:
                if log:
                    print("Inference outputs loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                sys.exit()
        else:
            print(f"ERROR! Output file does not exist!")
            sys.exit()

        if 'judge' in output:
            start_index = len(output['judge'])
        else:
            output['judge'] = []
            start_index = 0
        
        if log:
            print(f"Starting judge from claim with index: {start_index}.")
            
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        # Set up intermediate writing if user wants logs
        def on_tok(token):
            print(self.color_text(token, '96'), end="") # colored light cyan

        if not log:
            on_tok = None

        # Main judge loop
        for i, query in enumerate(input_json[start_index:]):
            if self.task == "fc":
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['claim']}", "93"))

                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query_judge(query['claim'], [query[evidence_split]], output_json['outputs'][i+start_index]['response'], query['label'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query_judge(query['claim'], query[evidence_split], output_json['outputs'][i+start_index]['response'], query['label'], on_tok=on_tok)
            else:
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['question']}", "93"))
                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query_judge(query['question'], query[evidence_split], output_json['outputs'][i+start_index]['response'], query['answer'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query_judge(query['question'], query[evidence_split], output_json['outputs'][i+start_index]['response'], query['answer'], on_tok=on_tok)
            
            output['judge'].append({'response': response, 'total_time': total_time})

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
            print(f"Judge processing finished in {total_time} seconds...")

        return output
    
    def validate_judge(self, input_json_file: str, output_json_file: str, log: bool=True):
        """
        Main function for judge validation processing.

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
        """
        if log:
            print(f"Starting judge validation...")

        with open(input_json_file, "r") as f:
            input_json = json.load(f)
        
        with open(output_json_file, "r") as f:
            output_json = json.load(f)

        if "judge" not in output_json:
            print("ERROR: no LLM-as-a-judge returns to be parsed")
            return

        n_invalid, n_correct, n_incorrect, total, incorrect, caused = self.validation_handler.parse_judge_output(output_json, log=log)

        output_json['judge_validation'] = {
            'n_invalid': n_invalid,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'total': total,
            'incorrect': incorrect,
            'caused': caused,
        }

        try:
            with open(output_json_file, "w") as f:
                json.dump(output_json, f)
        except Exception as e:
            print(f"ERROR with saving judge validation results, will revert to previous. {e}")
            output_json['judge_validation'] = {}
            with open(output_json_file, "w") as f:
                json.dump(output_json, f)

        if log:
            print(f"Finished judge validation.")


    def judge_new(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=20, golden: bool=False, evidence_split: str="doc"):
        """
        Main function for LLM-as-a-judge processing

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
            checkpoint (int): how often to save to the output file
            golden (bool): whether to use golden evidence
            evidence_split (str): which evidence split to use
        """
        if log:
            start_time = time.time()
            print(f"Starting {self.task} JUDGE NEW processing...")

        # Load outputs and previous checkpoint
        if os.path.isfile(output_json_file):
            with open(output_json_file, "r") as f:
                output_json = json.load(f)

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and 'golden' in output_json and 'evidence_split' in output_json and output_json['input_json_file'] == input_json_file and output_json['golden'] == golden and output_json['evidence_split'] == evidence_split:
                if log:
                    print("Inference outputs loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                sys.exit()
        else:
            print(f"ERROR! Output file does not exist!")
            sys.exit()

        if 'judge_new' in output:
            start_index = len(output['judge_new'])
        else:
            output['judge_new'] = []
            start_index = 0
        
        if log:
            print(f"Starting judge_new from claim with index: {start_index}.")
            
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        # Set up intermediate writing if user wants logs
        def on_tok(token):
            print(self.color_text(token, '96'), end="") # colored light cyan

        if not log:
            on_tok = None

        # Main judge loop
        for i, query in enumerate(input_json[start_index:]):
            if self.task == "fc":
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['claim']}", "93"))

                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query_judge_new(query['claim'], [query[evidence_split]], output_json['outputs'][i+start_index]['response'], query['label'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query_judge_new(query['claim'], query[evidence_split], output_json['outputs'][i+start_index]['response'], query['label'], on_tok=on_tok)
            else:
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['question']}", "93"))
                if golden: # if supposed to use the golden evidences
                    response, total_time = self.process_query_judge_new(query['question'], query[evidence_split], output_json['outputs'][i+start_index]['response'], query['answer'], on_tok=on_tok)
                else:
                    response, total_time = self.process_query_judge_new(query['question'], query[evidence_split], output_json['outputs'][i+start_index]['response'], query['answer'], on_tok=on_tok)
            
            output['judge_new'].append({'response': response, 'total_time': total_time})

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
            print(f"Judge new processing finished in {total_time} seconds...")

        return output

    def verify(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=50, golden: bool=False, evidence_split: str="doc", verifier_config_path: str="verifier/config.toml"):
        """
        Main function for Verifier processing

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
            checkpoint (int): how often to save to the output file
            golden (bool): whether to use golden evidence
            evidence_split (str): which evidence split to use
        """
        think_tag_start = "<think>"
        think_tag_end = "</think>"
        
        if log:
            start_time = time.time()
            print(f"Starting {self.task} VERIFIER processing...")

        # Load outputs and previous checkpoint
        if os.path.isfile(output_json_file):
            with open(output_json_file, "r") as f:
                output_json = json.load(f)

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and 'golden' in output_json and 'evidence_split' in output_json and output_json['input_json_file'] == input_json_file and output_json['golden'] == golden and output_json['evidence_split'] == evidence_split:
                if log:
                    print("Inference outputs loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                sys.exit()
        else:
            print(f"ERROR! Output file does not exist!")
            sys.exit()

        # Load verifier
        verifier = VerifierHandler(verifier_config_path)

        if 'verifier' in output:
            if verifier_config_path in output['verifier']:
                start_index = len(output['verifier'][verifier_config_path])
            else:
                output['verifier'][verifier_config_path] = list()
                start_index = 0
        else:
            output['verifier'] = {verifier_config_path: []}
            start_index = 0
        
        if log:
            print(f"Starting verifier from claim with index: {start_index}.")
            
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        # Load output dicts
        if self.task == "fc":
            final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output, log=False)
            n_valid, cf_matrix, accuracy, weighted_f1, macro_f1, per_class_f1s, per_taxonomy_f1s, final_output_dicts = self.validation_handler.validate_classification(input_json, final_output_dicts, log=False)
        else:
            final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output, log=False)
            n_valid, em_accuracy, n_invalid_ground_truth, final_output_dicts = self.validation_handler.validate_exact_match(input_json, final_output_dicts, log=False)


        # Main verifier loop
        for i, (query, response_dict, output_dict) in enumerate(zip(input_json[start_index:], output['outputs'][start_index:], final_output_dicts[start_index:])):
            # Find thinking
            lthink = response_dict['response'].find(think_tag_start)
            rthink = response_dict['response'].rfind(think_tag_end)

            if lthink == -1 and rthink == -1: # No thinking exists at all
                thinking = None
            else:
                if lthink == -1: # if left think tag doesn't exist, set it to first char
                    lthink = 0
                else:
                    lthink = lthink + len(think_tag_start)
        
                if rthink == -1: # if right think tag doesn't exist, set it to last char
                    rthink = len(response_dict['response'])
        
                thinking = response_dict['response'][lthink:rthink].strip()
            
            if self.task == "fc":
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['claim']}", "93"))                
                input_text = f"""Claim: {query['claim']}
Verdict: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}"""
                
                input_text_thinking = f"""Claim: {query['claim']}
Verdict: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}
Thinking: {thinking if thinking else 'UNKNOWN'}"""
            else:
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['question']}", "93"))
                input_text = f"""Question: {query['question']}
Final Answer: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}"""
                
                input_text_thinking = f"""Question: {query['question']}
Final Answer: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}
Thinking: {thinking if thinking else 'UNKNOWN'}"""
       
            t0 = time.time()
            score = verifier.score_text(input_text)
            score_thinking = verifier.score_text(input_text_thinking)
            t1 = time.time()
            output['verifier'][verifier_config_path].append({'score': score, 'score_thinking': score_thinking, 'total_time': t1-t0})

            if log:
                print(self.color_text(f"\nScores: {score:0.2f} {score_thinking:0.2f}\n", "93"))
                print(self.color_text(f"\nTotal time: {t1 - t0}\n", "93"))

            if i % checkpoint == 0:
                with open(output_json_file, "w") as f:
                    json.dump(output, f)
                print(self.color_text(f"\nSaved checkpoint!\n", "93"))

        with open(output_json_file, "w") as f:
            json.dump(output, f)

        if log:
            total_time = time.time() - start_time
            print(f"Verifier processing finished in {total_time} seconds...")

        return output


    def verify_multitask(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=50, golden: bool=False, evidence_split: str="doc", verifier_config_path: str="verifier/config_finqa_multitask.toml"):
        """
        Main function for Verifier processing

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
            checkpoint (int): how often to save to the output file
            golden (bool): whether to use golden evidence
            evidence_split (str): which evidence split to use
        """
        think_tag_start = "<think>"
        think_tag_end = "</think>"
        
        if log:
            start_time = time.time()
            print(f"Starting {self.task} MULTI-TASK VERIFIER processing...")

        # Load outputs and previous checkpoint
        if os.path.isfile(output_json_file):
            with open(output_json_file, "r") as f:
                output_json = json.load(f)

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and 'golden' in output_json and 'evidence_split' in output_json and output_json['input_json_file'] == input_json_file and output_json['golden'] == golden and output_json['evidence_split'] == evidence_split:
                if log:
                    print("Inference outputs loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                sys.exit()
        else:
            print(f"ERROR! Output file does not exist!")
            sys.exit()

        # Load verifier
        verifier = MultiTaskVerifierHandler(verifier_config_path)

        if 'verifier' in output:
            if verifier_config_path in output['verifier']:
                start_index = len(output['verifier'][verifier_config_path])
            else:
                output['verifier'][verifier_config_path] = list()
                start_index = 0
        else:
            output['verifier'] = {verifier_config_path: []}
            start_index = 0
        
        if log:
            print(f"Starting verifier from claim with index: {start_index}.")
            
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        # Load output dicts
        if self.task == "fc":
            final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output, log=False)
            n_valid, cf_matrix, accuracy, weighted_f1, macro_f1, per_class_f1s, per_taxonomy_f1s, final_output_dicts = self.validation_handler.validate_classification(input_json, final_output_dicts, log=False)
        else:
            final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output, log=False)
            n_valid, em_accuracy, n_invalid_ground_truth, final_output_dicts = self.validation_handler.validate_exact_match(input_json, final_output_dicts, log=False)


        # Main verifier loop
        for i, (query, response_dict, output_dict) in enumerate(zip(input_json[start_index:], output['outputs'][start_index:], final_output_dicts[start_index:])):
            # Find thinking
            lthink = response_dict['response'].find(think_tag_start)
            rthink = response_dict['response'].rfind(think_tag_end)

            if lthink == -1 and rthink == -1: # No thinking exists at all
                thinking = None
            else:
                if lthink == -1: # if left think tag doesn't exist, set it to first char
                    lthink = 0
                else:
                    lthink = lthink + len(think_tag_start)
        
                if rthink == -1: # if right think tag doesn't exist, set it to last char
                    rthink = len(response_dict['response'])
        
                thinking = response_dict['response'][lthink:rthink].strip()
            
            if self.task == "fc":
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['claim']}", "93"))                
                input_text = f"""Claim: {query['claim']}
Verdict: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}"""
                
                input_text_thinking = f"""Claim: {query['claim']}
Verdict: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}
Thinking: {thinking if thinking else 'UNKNOWN'}"""
            else:
                if log:
                    print(self.color_text(f"{i+start_index} Processing: {query['question']}", "93"))
                input_text = f"""Question: {query['question']}
Final Answer: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}"""
                
                input_text_thinking = f"""Question: {query['question']}
Final Answer: {output_dict['label'] if (output_dict['label'] and output_dict['label'] != 'None') else 'UNKNOWN'}
Justification: {output_dict['justification'] if output_dict['justification'] else 'UNKNOWN'}
Thinking: {thinking if thinking else 'UNKNOWN'}"""
       
            t0 = time.time()
            score, scoping_score = verifier.score_text(input_text)
            score_thinking, scoping_score_thinking = verifier.score_text(input_text_thinking)
            t1 = time.time()
            output['verifier'][verifier_config_path].append({'score': score, 'score_thinking': score_thinking, 'scoping_score': scoping_score, 'scoping_score_thinking': scoping_score_thinking, 'total_time': t1-t0})

            if log:
                print(self.color_text(f"\nScores: {score:0.2f} {score_thinking:0.2f}\n", "93"))
                print(self.color_text(f"\nScoping Scores: {scoping_score:0.2f} {scoping_score_thinking:0.2f}\n", "93"))
                print(self.color_text(f"\nTotal time: {t1 - t0}\n", "93"))

            if i % checkpoint == 0:
                with open(output_json_file, "w") as f:
                    json.dump(output, f)
                print(self.color_text(f"\nSaved checkpoint!\n", "93"))

        with open(output_json_file, "w") as f:
            json.dump(output, f)

        if log:
            total_time = time.time() - start_time
            print(f"Verifier processing finished in {total_time} seconds...")

        return output
