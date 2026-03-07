import os
import sys
import json
import time
from typing import List, Dict, Any
from utils.corenlp_quantity_segmenter import Segmenter
from llm.prompt_handler import PromptHandler
from llm.llm_handler import LLMHandler



class ScopingErrorMeasurer:
    """
    Class that manages measuring scoping errors.
    
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
            self.llm_handler = LLMHandler(completion_params_file="llm/completion_params_judge.json")
        else:
            from llm.llm_handler_delftblue import LLMHandler
            self.llm_handler = LLMHandler()

        self.prompt_handler = PromptHandler(task=task)
        self.segmenter = Segmenter()

        self.think_tag_start = "<think>"
        self.think_tag_end = "</think>"
        self.final_answer_tag = "[Label]" if self.task == "fc" else "[Final Answer]"
        self.justification_tag = "[Justification]"

        self.extractive_check_system_prompt = self.prompt_handler.get_system_message(variant="extractive_check")
        self.non_extractive_check_system_prompt = self.prompt_handler.get_system_message(variant="non_extractive_check")
        self.importance_check_system_prompt = self.prompt_handler.get_system_message(variant="importance_check")


    def color_text(self, text: str, color_code: str) -> str:
        """Function used to color cmd line text."""
        return f"\033[{color_code}m{text}\033[0m"
    
    # def process_query(self, query: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]], on_tok=None) -> str:
    #     """Runs a single fact-check/qa query."""
    #     user_prompt = self.prompt_handler.get_user_message(query, evidence_list, fc_example_list)
    #     response, total_time = self.llm_handler.generate_response(self.system_prompt, user_prompt, on_token=on_tok)
    #     return response, total_time
    
    # def process_query_judge(self, query: str, evidence_list: List[str], response: str, actual_label: str, on_tok=None) -> str:
    #     """Runs a single fact-check/qa LLM-as-a-judge query."""
    #     user_prompt = self.prompt_handler.get_user_message(query, evidence_list, fc_example_list=None, metadata={'response': response, 'actual_label': actual_label}, variant="judge")
    #     response, total_time = self.llm_handler.generate_response(self.judge_system_prompt, user_prompt, on_token=on_tok)
    #     return response, total_time
    
    def set_up_input_json_quantities(self, input_json_file: str, input_json_file_quantities: str, evidence_split: str, checkpoint: int=20, log: bool = False):
        """
        Function to set up inputs file. Will extract quantities if not complete.

        Args:
            input_json_file (str): file path to input file
            input_json_file_quantities (str): file path to input quantities file
            evidence_split (str): which of the input evidences to extract from
            checkpoint (int): how often to save to the output file
            log (bool): whether to log intermediate statements
        """
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        if not input_json_file_quantities: # make file path if not given
            input_json_file_quantities = f"{input_json_file[:-5]}_quantities.json"
        
        if not os.path.isfile(input_json_file_quantities): # create file if it doesn't exist
            input_json_quantities = {evidence_split: []}
            with open(input_json_file_quantities, "w") as f:
                json.dump(input_json_quantities, f)
            
        # Load quantities file
        with open(input_json_file_quantities, "r") as f:
            input_json_quantities = json.load(f)

        # Check if it is already complete
        if not isinstance(input_json_quantities, dict):
            input_json_quantities = {evidence_split: []}
            start_index = 0
            print(f"Input quantities json is nonexistent. Will generate quantities for {evidence_split} split now.")
        else:
            if evidence_split in input_json_quantities: 
                if len(input_json_quantities[evidence_split]) == len(input_json):
                    print(f"Input quantities json is complete with length {len(input_json_quantities)} for split {evidence_split}")
                    return input_json_file_quantities
                else:
                    start_index = len(input_json_quantities[evidence_split])
                    print(f"Input quantities json is incomplete with length {len(input_json_quantities)} for split {evidence_split}. Will generate rest of quantities now.")
            else:
                input_json_quantities[evidence_split] = []
                start_index = 0
                print(f"Input quantities json has no split {evidence_split}. Will generate rest of quantities now.")

        start_time = time.time()
        for i, query in enumerate(input_json[start_index:]):
            if log:
                print(self.color_text(f"{i+start_index} Processing: {query['claim'] if self.task == 'fc' else query['question']}", "93"))

            # extract input quantities
            final_quantities = dict()
            claim_text = self.segmenter.normalize_text(query['claim']) if self.task == 'fc' else self.segmenter.normalize_text(query['question'])
            evidences_text = [query[evidence_split]] if evidence_split == "doc" and self.task == "fc" else [l for l in query[evidence_split]]
            if self.task == "qa" and evidence_split == "doc":
                table = [s.split("|") for s in evidences_text[1].split("\n")[1:]]
                evidences_text[1] = "\n".join([self.segmenter.convert_finqa_table_row_to_text(table[0], row) for row in table[1:]])
            evidences_text = [self.segmenter.normalize_text(evidence) for evidence in evidences_text]
            
            # Extract claim quantities
            _, _, claim_quantity_natural_language = self.segmenter.segment(claim_text)
            for output in claim_quantity_natural_language:
                if output['quantity'] not in final_quantities:
                    
                    final_quantities[output['quantity']] = [{'quantity_texts': [output['quantity_text']], 'sentence_idxs': [f"claim:{output['sentence_idx']}"], 'sentence_texts': [output['sentence_text']], 'quantity_metadata': self.segmenter.normalize_quantity_phrase(output['quantity_text'])}]
                else:
                    # for each quantity type, compute agreement between quantity texts to ensure they measure about the same thing?
                    quantity_added = False
                    second = self.segmenter.normalize_quantity_phrase(output['quantity_text'])
                    for subquantity in final_quantities[output['quantity']]:
                        first = subquantity['quantity_metadata']
                        same_nums = self.segmenter.typed_quantity_similarity(first, second)
                        if same_nums:
                            quantity_added = True
                            if len(output['quantity_text']) < len(subquantity['quantity_texts'][0]):
                                subquantity['quantity_texts'].insert(0, output['quantity_text'])
                                subquantity['sentence_idxs'].insert(0, f"claim:{output['sentence_idx']}")
                                subquantity['sentence_texts'].insert(0, output['sentence_text'])
                                subquantity['quantity_metadata'] = second
                            else:
                                subquantity['quantity_texts'].append(output['quantity_text'])
                                subquantity['sentence_idxs'].append(f"claim:{output['sentence_idx']}")
                                subquantity['sentence_texts'].append(output['sentence_text'])
                    
                    if not quantity_added:
                        final_quantities[output['quantity']].append({'quantity_texts': [output['quantity_text']], 'sentence_idxs': [f"claim:{output['sentence_idx']}"], 'sentence_texts': [output['sentence_text']], 'quantity_metadata': second})

            for j, evidence_text in enumerate(evidences_text):
                _, _, evidence_quantity_natural_language = self.segmenter.segment(evidence_text)

                for output in evidence_quantity_natural_language:
                    if output['quantity'] not in final_quantities:
                        
                        final_quantities[output['quantity']] = [{'quantity_texts': [output['quantity_text']], 'sentence_idxs': [f"evidence{j}:{output['sentence_idx']}"], 'sentence_texts': [output['sentence_text']], 'quantity_metadata': self.segmenter.normalize_quantity_phrase(output['quantity_text'])}]
                    else:
                        # for each quantity type, compute agreement between quantity texts to ensure they measure about the same thing?
                        quantity_added = False
                        second = self.segmenter.normalize_quantity_phrase(output['quantity_text'])
                        for subquantity in final_quantities[output['quantity']]:
                            first = subquantity['quantity_metadata']
                            same_nums = self.segmenter.typed_quantity_similarity(first, second)
                            if same_nums:
                                quantity_added = True
                                if len(output['quantity_text']) < len(subquantity['quantity_texts'][0]):
                                    subquantity['quantity_texts'].insert(0, output['quantity_text'])
                                    subquantity['sentence_idxs'].insert(0, f"evidence{j}:{output['sentence_idx']}")
                                    subquantity['sentence_texts'].insert(0, output['sentence_text'])
                                    subquantity['quantity_metadata'] = second
                                else:
                                    subquantity['quantity_texts'].append(output['quantity_text'])
                                    subquantity['sentence_idxs'].append(f"evidence{j}:{output['sentence_idx']}")
                                    subquantity['sentence_texts'].append(output['sentence_text'])

            if log:
                print(self.color_text(f"Numbers gotten: {list(final_quantities.keys())}", "91"))

            input_json_quantities[evidence_split].append(final_quantities)

            if i % checkpoint == 0:
                with open(input_json_file_quantities, "w") as f:
                    json.dump(input_json_quantities, f)
                print(self.color_text(f"\nSaved checkpoint!\n", "93"))

        with open(input_json_file_quantities, "w") as f:
            json.dump(input_json_quantities, f)

        if log:
            total_time = time.time() - start_time
            print(f"Input quantity segmentation processing finished in {total_time} seconds...")

        return input_json_file_quantities


    def extract_thinking_quantities(self, thinking):
        """
        Extract thinking quantities given query.

        Args:
            thinking (str): thinking for that query

        Returns:
            final_quantities (Dict[str, Any]): quantities found in that response
        """
        text = self.segmenter.normalize_text(thinking)
        _, _, quantity_natural_language = self.segmenter.segment(text)
        final_quantities = dict()
        for output in quantity_natural_language:
            if output['quantity'] not in final_quantities:
                
                final_quantities[output['quantity']] = [{'quantity_texts': [output['quantity_text']], 'sentence_idxs': [output['sentence_idx']], 'sentence_texts': [output['sentence_text']], 'quantity_metadata': self.segmenter.normalize_quantity_phrase(output['quantity_text'])}]
            else:
                # for each quantity type, compute agreement between quantity texts to ensure they measure about the same thing?
                quantity_added = False
                second = self.segmenter.normalize_quantity_phrase(output['quantity_text'])
                for subquantity in final_quantities[output['quantity']]:
                    first = subquantity['quantity_metadata']
                    same_nums = self.segmenter.typed_quantity_similarity(first, second)
                    if same_nums:
                        quantity_added = True
                        if len(output['quantity_text']) < len(subquantity['quantity_texts'][0]):
                            subquantity['quantity_texts'].insert(0, output['quantity_text'])
                            subquantity['sentence_idxs'].insert(0, output['sentence_idx'])
                            subquantity['sentence_texts'].insert(0, output['sentence_text'])
                            subquantity['quantity_metadata'] = second
                        else:
                            subquantity['quantity_texts'].append(output['quantity_text'])
                            subquantity['sentence_idxs'].append(output['sentence_idx'])
                            subquantity['sentence_texts'].append(output['sentence_text'])
                
                if not quantity_added:
                    final_quantities[output['quantity']].append({'quantity_texts': [output['quantity_text']], 'sentence_idxs': [output['sentence_idx']], 'sentence_texts': [output['sentence_text']], 'quantity_metadata': second})
                
        return final_quantities


    def process_output(self, response):
        """
        Function to split up parts of response

        Args:
            response (str): response of fact-check or QA LLM

        Returns:
            thinking (str)
            final_answer (str)
            justification (str)
        """
        lthink = response.find(self.think_tag_start)
        rthink = response.rfind(self.think_tag_end)

        if lthink == -1: # if left think tag doesn't exist, set it to first char
            lthink = 0
        else:
            lthink = lthink + len(self.think_tag_start)

        if rthink == -1: # if right think tag doesn't exist, set it to last char
            rthink = len(response)

        thinking = response[lthink:rthink].strip()
            
        rfinal_answer = response.rfind(self.final_answer_tag)
        rjustification = response.rfind(self.justification_tag)

        if rjustification == -1:
            rjustification = len(response)
            justification = ""
        else:
            justification = response[rjustification + len(self.justification_tag):]
            if justification[0] == ":":
                justification = justification[1:].strip()

        if rfinal_answer == -1:
            final_answer = ""
        else:
            final_answer = response[rfinal_answer + len(self.final_answer_tag):rjustification].strip()
            if len(final_answer) > 2 and final_answer[0] == ":":
                final_answer = final_answer[1:].strip()

        return thinking, final_answer, justification


    def extractive_check(self, query, evidences, thinking, thinking_quantities, formatted_quantities, log, on_tok):
        """
        Function that checks whether a list of extractive quantities is extracted correctly.

        Args:
            query (str): claim to fact-check or question to answer
            evidences (List[str]): list of input evidences
            thinking (str): string thinking
            thinking_quantities (Dict[str, Any]): dictionary of thinking quantities
            formatted_quantities (List[str]): list of statistics in string format
            log (bool): whether to print intermediate statements or not
            on_tok (Callable): function that can print intermediate LLM outputs

        Returns:
            response_dict (Dict[str, Any]): dictionary containing extractive response information
            thinking_quantities (Dict[str, Any]): thinking quantity dict with extractive info included
        """
        response_dict = {"str_response": "{\"results\": []}", "time": 0.0, "results": [], "json": {"results": []}, "valid_json": True}
        if len(formatted_quantities) == 0:
            return response_dict

        user_prompt = self.prompt_handler.get_user_message(
            query=query, 
            evidence_list=evidences, 
            fc_example_list=None, 
            metadata={"statistics": formatted_quantities, "response": thinking}, 
            variant="extractive_check")  
        response, t = self.llm_handler.generate_response(self.extractive_check_system_prompt, user_prompt, on_token=on_tok)

        response_dict['str_response'] = response
        response_dict['time'] = t

        valid_json = True
        try:
            p_json = response[response.index('{'): response.rfind('}')+1]
            j = json.loads(p_json)
            response_dict['json'] = j
        except Exception as e:
            if log:
                print(f"ERROR: parsing json response {response}")
            valid_json = False

        response_dict['valid_json'] = valid_json

        if valid_json:
            try:
                if len(formatted_quantities) != len(j['results']): # value-based parsing
                    if log:
                        print(f"WARNING: json missing checking some quantities! {j} {formatted_quantities}")
                    for result in j['results']:
                        q = result['statistic_quantity'] if isinstance(result['statistic_quantity'], str) else str(result['statistic_quantity'])
                        text = result['statistic_text']
                        correct = result['correct']
                        for t_quantity_type in thinking_quantities[q]:
                            if text in t_quantity_type['quantity_texts']:
                                t_quantity_type['correct'] = correct
                                t_quantity_type['check'] = result
                                response_dict['results'].append(correct)
                                break
                        
                else: # length-based parsing
                    ctr = 0
                    for t_quantity in thinking_quantities:
                        for t_quantity_type in thinking_quantities[t_quantity]:
                            if t_quantity_type["type"] == "extractive":
                                result = j['results'][ctr]
                                q = result['statistic_quantity'] if isinstance(result['statistic_quantity'], str) else str(result['statistic_quantity'])
                                if q != t_quantity:
                                    if log:
                                        print(f"WARNING: during length parsing json quantities not equal! {result} {t_quantity_type}")
                                correct = result['correct']
                                t_quantity_type['correct'] = result['correct']
                                t_quantity_type['check'] = result
                                response_dict['results'].append(correct)
                                ctr += 1

            except Exception as e:
                valid_json = False
                response_dict['valid_json'] = False
                if log:
                    print(f"ERROR parsing: {e} with json {response}")

        if log:
            print(f"\n\nExtractive Results:")
            print(f"Quantities ({len(formatted_quantities)}): {formatted_quantities};\nResults ({len(response_dict['results'])}): {response_dict['results']}")

        return response_dict, thinking_quantities
    

    def non_extractive_check(self, query, evidences, thinking, thinking_quantities, formatted_quantities, log, on_tok):
        """
        Function that checks whether a list of non extractive quantities are parametric or extrapolative and correct.

        Args:
            query (str): claim to fact-check or question to answer
            evidences (List[str]): list of input evidences
            thinking (str): string thinking
            thinking_quantities (Dict[str, Any]): dictionary of thinking quantities
            formatted_quantities (List[str]): list of statistics in string format
            log (bool): whether to print intermediate statements or not
            on_tok (Callable): function that can print intermediate LLM outputs

        Returns:
            response_dict (Dict[str, Any]): dictionary containing non extractive response information
            thinking_quantities (Dict[str, Any]): thinking quantity dict with non extractive info included
        """
        response_dict = {"str_response": "{\"results\": []}", "time": 0.0, "results": [], "types": [], "json": {"results": []}, "valid_json": True}
        if len(formatted_quantities) == 0:
            return response_dict

        user_prompt = self.prompt_handler.get_user_message(
            query=query, 
            evidence_list=evidences, 
            fc_example_list=None, 
            metadata={"statistics": formatted_quantities, "response": thinking}, 
            variant="non_extractive_check")  
        response, t = self.llm_handler.generate_response(self.non_extractive_check_system_prompt, user_prompt, on_token=on_tok)

        response_dict['str_response'] = response
        response_dict['time'] = t

        valid_json = True
        try:
            p_json = response[response.index('{'): response.rfind('}')+1]
            j = json.loads(p_json)
            response_dict['json'] = j
        except Exception as e:
            if log:
                print(f"ERROR: parsing json response {response}")
            valid_json = False

        response_dict['valid_json'] = valid_json

        if valid_json:
            try:
                if len(formatted_quantities) != len(j['results']): # value-based parsing
                    if log:
                        print(f"WARNING: json missing checking some quantities! {j} {formatted_quantities}")
                    for result in j['results']:
                        q = result['statistic_quantity'] if isinstance(result['statistic_quantity'], str) else str(result['statistic_quantity'])
                        text = result['statistic_text']
                        correct = result['correct']
                        typ = result['quantity_type']
                        for t_quantity_type in thinking_quantities[q]:
                            if text in t_quantity_type['quantity_texts']:
                                t_quantity_type['correct'] = correct
                                t_quantity_type['check'] = result
                                t_quantity_type['type'] = typ
                                response_dict['results'].append(correct)
                                response_dict['types'].append(typ)
                                break
                        
                else: # length-based parsing
                    ctr = 0
                    for t_quantity in thinking_quantities:
                        for t_quantity_type in thinking_quantities[t_quantity]:
                            if t_quantity_type["type"] != "extractive":
                                result = j['results'][ctr]
                                q = result['statistic_quantity'] if isinstance(result['statistic_quantity'], str) else str(result['statistic_quantity'])
                                if q != t_quantity:
                                    if log:
                                        print(f"WARNING: during length parsing json quantities not equal! {result} {t_quantity_type}")
                                correct = result['correct']
                                typ = result['quantity_type']
                                t_quantity_type['correct'] = result['correct']
                                t_quantity_type['check'] = result
                                t_quantity_type['type'] = typ
                                response_dict['results'].append(correct)
                                response_dict['types'].append(typ)
                                ctr += 1

            except Exception as e:
                valid_json = False
                response_dict['valid_json'] = False
                if log:
                    print(f"ERROR parsing: {e} with json {response}")

        if log:
            print(f"\n\nNon-Extractive Results:")
            print(f"Quantities ({len(formatted_quantities)}): {formatted_quantities};\nResults ({len(response_dict['results'])}): {response_dict['results']}\n Types ({len(response_dict['types'])}): {response_dict['types']}")

        return response_dict, thinking_quantities
    

    def importance_check(self, query, evidences, thinking, final_answer, justification, thinking_quantities, formatted_quantities, log, on_tok):
        """
        Function that gets importance weights for statistics for scoring calculation.

        Args:
            query (str): claim to fact-check or question to answer
            evidences (List[str]): list of input evidences
            thinking (str): string thinking
            final_answer (str): string final answer
            justification (str): string justification
            thinking_quantities (Dict[str, Any]): dictionary of thinking quantities
            formatted_quantities (List[str]): list of statistics in string format
            log (bool): whether to print intermediate statements or not
            on_tok (Callable): function that can print intermediate LLM outputs

        Returns:
            response_dict (Dict[str, Any]): dictionary containing non extractive response information
            thinking_quantities (Dict[str, Any]): thinking quantity dict with non extractive info included
        """
        response_dict = {"str_response": "{\"results\": []}", "time": 0.0, "weights": [], "json": {"results": []}, "valid_json": True}
        if len(formatted_quantities) == 0:
            return response_dict

        user_prompt = self.prompt_handler.get_user_message(
            query=query, 
            evidence_list=evidences, 
            fc_example_list=None, 
            metadata={"statistics": formatted_quantities, "thinking": thinking, "predicted": final_answer, "justification": justification}, 
            variant="importance_check")  
        response, t = self.llm_handler.generate_response(self.importance_check_system_prompt, user_prompt, on_token=on_tok)

        response_dict['str_response'] = response
        response_dict['time'] = t

        valid_json = True
        try:
            p_json = response[response.index('{'): response.rfind('}')+1]
            j = json.loads(p_json)
            response_dict['json'] = j
        except Exception as e:
            if log:
                print(f"ERROR: parsing json response {response}")
            valid_json = False

        response_dict['valid_json'] = valid_json

        if valid_json:
            total_weight = 0.0 # for weight regularization later
            try:
                if len(formatted_quantities) != len(j['results']): # value-based parsing
                    if log:
                        print(f"WARNING: json missing weighing some quantities! {j} {formatted_quantities}")
                    for result in j['results']:
                        q = result['statistic_quantity'] if isinstance(result['statistic_quantity'], str) else str(result['statistic_quantity'])
                        text = result['statistic_text']
                        weight = float(result['importance_weight'])
                        for t_quantity_type in thinking_quantities[q]:
                            if text in t_quantity_type['quantity_texts']:
                                t_quantity_type['weight'] = weight
                                t_quantity_type['weight_check'] = result
                                response_dict['weights'].append(weight)
                                total_weight += weight
                                break
                        
                else: # length-based parsing
                    ctr = 0
                    for t_quantity in thinking_quantities:
                        for t_quantity_type in thinking_quantities[t_quantity]:
                            if t_quantity_type["type"] != "extractive":
                                result = j['results'][ctr]
                                q = result['statistic_quantity'] if isinstance(result['statistic_quantity'], str) else str(result['statistic_quantity'])
                                if q != t_quantity:
                                    if log:
                                        print(f"WARNING: during importance check length parsing json quantities not equal! {result} {t_quantity_type}")
                                weight = float(result['importance_weight'])
                                t_quantity_type['weight'] = weight
                                t_quantity_type['weight_check'] = result
                                response_dict['weights'].append(weight)
                                ctr += 1
                                total_weight += weight

                if total_weight != 1.0: # re-scale if total_weight is not 1
                    if log:
                        print(f"WARNING: weights not equal to 1, rescaling! {total_weight} {response_dict['weights']}")
                    multiplier = 1.0 / total_weight
                    for t_quantity in thinking_quantities:
                        for t_quantity_type in thinking_quantities[t_quantity]:
                            if isinstance(t_quantity_type['weight'], float):
                                t_quantity_type['weight'] *= multiplier
                    for j, weight in enumerate(response_dict['weights']):
                        response_dict['weights'][j] = weight * multiplier
            
            except Exception as e:
                valid_json = False
                response_dict['valid_json'] = False
                if log:
                    print(f"ERROR parsing: {e} with json {response}")

        if log:
            print(f"\n\nImportance Results:")
            print(f"Quantities ({len(formatted_quantities)}): {formatted_quantities};\nWeights ({len(response_dict['weights'])}): {response_dict['weights']}")

        return response_dict, thinking_quantities


    def run(self, input_json_file: str, output_json_file: str, log: bool=True, checkpoint: int=5, golden: bool=False, input_json_file_quantities: str = None):
        """
        Main function for measuring scoping error processing

        Args:
            input_json_file (str): json file that contains inputs
            output_json_file (str): json file to save outputs
            log (bool): whether to log intermediate statements
            checkpoint (int): how often to save to the output file
            golden (bool): whether to use golden evidence
            input_json_file_quantities (str): json file that contains the quantities for the inputs (optional)
        """
        if log:
            start_time = time.time()
            print(f"Starting {self.task} advanced judge processing...")

        # Load outputs and previous checkpoint
        if os.path.isfile(output_json_file):
            with open(output_json_file, "r") as f:
                output_json = json.load(f)

            if output_json and 'outputs' in output_json and 'input_json_file' in output_json and 'golden' in output_json and output_json['input_json_file'] == input_json_file and output_json['golden'] == golden:
                if log:
                    print("Inference outputs loaded.")
                output = output_json
            else:
                print(f"ERROR! Output file exists and has results not matching current input file!")
                sys.exit()
        else:
            print(f"ERROR! Output file does not exist!")
            sys.exit()

        if 'advanced_judge' in output:
            start_index = len(output['advanced_judge'])
        else:
            output['advanced_judge'] = []
            start_index = 0
        
        if log:
            print(f"Starting advanced judge from claim with index: {start_index}.")
            
        # Load input file
        with open(input_json_file) as f:
            input_json = json.load(f)

        # Load input quantities file
        evidence_split = "doc" if golden else "quantemp_evidences"
        input_json_file_quantities = self.set_up_input_json_quantities(input_json_file, input_json_file_quantities, evidence_split, checkpoint, log)
        with open(input_json_file_quantities) as f:
            input_json_quantities = json.load(f)

        # Set up intermediate writing if user wants logs
        def on_tok(token):
            print(self.color_text(token, '96'), end="") # colored light cyan

        if not log:
            on_tok = None

        # Main advanced judge loop
        for i, (query, input_quantities, outputs) in enumerate(zip(input_json[start_index:], input_json_quantities[evidence_split][start_index:], output_json['outputs'][start_index:])):
            if log:
                print(self.color_text(f"{i+start_index} Processing: {query['claim'] if self.task == 'fc' else query['question']}", "93"))
            
            t0 = time.time()

            q = query['claim'] if self.task == "fc" else query['question']
            evidences = [query[evidence_split]] if self.task == "fc" and evidence_split == "doc" else query[evidence_split]
            thinking, final_answer, justification = self.process_output(outputs['response'])
            
            # extract thinking quantities
            thinking_quantities = self.extract_thinking_quantities(thinking)
            if log:
                print(self.color_text(f"Thinking numbers gotten: {list(thinking_quantities.keys())}", "91"))

            # match thinking quantities to input quantities searching for extractive & non extractive
            extractive_formatted_quantities = [] # contains extractive quantities
            non_extractive_formatted_quantities = [] # contains parametric/extrapolative quantities
            formatted_quantities = [] # contains all quantities
            extractive_ctr = 1
            non_extractive_ctr = 1
            ctr = 1

            for t_quantity in thinking_quantities:
                for t_quantity_type in thinking_quantities[t_quantity]:
                    t_quantity_type["type"] = "unknown" # track whether statistic was extracted, extrapolated, or parametric
                    t_quantity_type["correct"] = "unknown" # track whether statistic was extracted, extrapolated, or parametric correctly
                    t_quantity_type['check'] = {} # place for the outputs for the metric
                    t_quantity_type['weight'] = None # track weight of this statistic on output
                    t_quantity_type['weight_check'] = {} # place for weighting outputs
                
                if t_quantity in input_quantities: # if exact match of quantity in input dict
                    for i, t_quantity_type in enumerate(thinking_quantities[t_quantity]):
                        # Check if matches with any stats in input
                        for i_quantity_type in input_quantities[t_quantity]:
                            if self.segmenter.typed_quantity_similarity(t_quantity_type['quantity_metadata'], i_quantity_type['quantity_metadata']):
                                # Mark as extractive!
                                t_quantity_type["type"] = "extractive"                                

                        if t_quantity_type["type"] == "extractive":
                            extractive_formatted_quantities.append(f"STATISTIC ID: {extractive_ctr}\n- quantity: {t_quantity}\n- text: '{t_quantity_type['quantity_texts'][0]}'")
                            extractive_ctr += 1
                            formatted_quantities.append(f"STATISTIC ID: {ctr}\n- quantity: {t_quantity}\n- text: '{t_quantity_type['quantity_texts'][0]}'")
                            ctr += 1
                        else:
                            non_extractive_formatted_quantities.append(f"STATISTIC ID: {non_extractive_ctr}\n- quantity: {t_quantity}\n- text: '{t_quantity_type['quantity_texts'][0]}'")
                            non_extractive_ctr += 1
                            formatted_quantities.append(f"STATISTIC ID: {ctr}\n- quantity: {t_quantity}\n- text: '{t_quantity_type['quantity_texts'][0]}'")
                            ctr += 1
        
                else:
                    # mark whole group as non extractive
                    for i, t_quantity_type in enumerate(thinking_quantities[t_quantity]):
                        non_extractive_formatted_quantities.append(f"STATISTIC ID: {non_extractive_ctr}\n- quantity: {t_quantity}\n- text: '{t_quantity_type['quantity_texts'][0]}'")
                        non_extractive_ctr += 1
                        formatted_quantities.append(f"STATISTIC ID: {ctr}\n- quantity: {t_quantity}\n- text: '{t_quantity_type['quantity_texts'][0]}'")
                        ctr += 1

            # Now we have a list of extractive, nonextractive, and overall statistics
            # Do extractive check
            extractive_response_dict, thinking_quantities = self.extractive_check(q, evidences, thinking, thinking_quantities, extractive_formatted_quantities, log, on_tok)

            # Do nonextractive check
            non_extractive_response_dict, thinking_quantities = self.non_extractive_check(q, evidences, thinking, thinking_quantities, non_extractive_formatted_quantities, log, on_tok)

            # Do importance check
            importance_response_dict, thinking_quantities = self.importance_check(q, evidences, thinking, final_answer, justification, thinking_quantities, formatted_quantities, log, on_tok)


            # Aggregate results
            scoping_error_dict = {
                'valid': True if extractive_response_dict['valid_json'] and non_extractive_response_dict['valid_json'] and importance_response_dict['valid_json'] else False,
                'scoping_error': False,
                'score': 0.0, 
                'per_class': {
                    'extractive': {'n': 0, 'n_correct': 0, 'n_incorrect': 0, 'n_unknown': 0},
                    'extrapolative': {'n': 0, 'n_correct': 0, 'n_incorrect': 0, 'n_unknown': 0},
                    'parametric': {'n': 0, 'n_correct': 0, 'n_incorrect': 0, 'n_unknown': 0},
                    'unknown': {'n': 0, 'n_correct': 0, 'n_incorrect': 0, 'n_unknown': 0},
                },
                'errors': [],
                'unknowns': [],
                'thinking_quantities': thinking_quantities,
                'response_dicts': {
                    'extractive': extractive_response_dict,
                    'non_extractive': non_extractive_response_dict,
                    'importance': importance_response_dict,
                }
            }

            try:
                for t_quantity in thinking_quantities:
                    for j, t_quantity_type in enumerate(thinking_quantities[t_quantity]):
                        if t_quantity_type['type'] in  scoping_error_dict['per_class']:
                            typ = t_quantity_type['type']
                        else:
                            typ = "unknown"

                        scoping_error_dict['per_class'][typ]['n'] += 1
                        if t_quantity_type['correct'] == 'true':
                            scoping_error_dict['per_class'][typ]['n_correct'] += 1
                            if t_quantity_type['weight'] and isinstance(t_quantity_type['weight'], float):
                                scoping_error_dict['score'] += t_quantity_type['weight']
                        elif t_quantity_type['correct'] == 'false':
                            scoping_error_dict['per_class'][typ]['n_incorrect'] += 1
                            # don't add anything to score
                            scoping_error_dict['scoping_error'] = True
                            scoping_error_dict['errors'].append({'quantity': t_quantity, 'quantity_i': j, 'type': t_quantity_type['type'], 'correct': t_quantity_type['correct'], 'weight': t_quantity_type['weight'], 'check': t_quantity_type['check']})
                        else:
                            scoping_error_dict['per_class'][typ]['n_unknown'] += 1
                            if t_quantity_type['weight'] and isinstance(t_quantity_type['weight'], float):
                                scoping_error_dict['score'] += t_quantity_type['weight'] * 0.5
                            scoping_error_dict['unknowns'].append({'quantity': t_quantity, 'quantity_i': j, 'type': t_quantity_type['type'], 'correct': t_quantity_type['correct'], 'weight': t_quantity_type['weight'], 'check': t_quantity_type['check']})
            except Exception as e:
                if log:
                    print(f"ERROR: getting scoping statistics: {e}")
                scoping_error_dict['scoping_error'] = None
                scoping_error_dict['valid'] = False
            output['advanced_judge'].append(scoping_error_dict)

            if log:
                print(self.color_text(f"\nTotal time: {time.time() - t0}\n", "93"))

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