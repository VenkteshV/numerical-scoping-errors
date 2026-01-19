import re
import json
from math import isclose
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

class ValidationHandler:
    """
    Class that does different types of validation for outputs.
    
    Internal States:
    - task: "fc" or "qa"
    """
    def __init__(
            self,
            task: str = "fc"):

        self.task = task
        if task == "fc":
            self.labels = ['True', 'False', 'Conflicting']
            self.label_map = {"SUPPORTS": "True", "REFUTES": "False", "CONFLICTING": "Conflicting"}
            self.separators = ["[Label]:", "[Justification]:"]
            self.taxonomy_label_set = ['statistical', 'temporal', 'interval', 'comparison']
        else:
            self.labels = ["yes", "no"]
            self.label_map = {"YES": "yes", "NO": "no"}
            self.separators = ["[Final Answer]:", "[Justification]:"]
            self.taxonomy_label_set = []

    def _clean_num(self, text: str):
        """
        Function to remove erroneous characters from number reponses

        Args:
            text (str): text to remove characters from

        Returns:
            text (str): cleaned text
        """
        text = text.replace("$", "")
        text = text.replace("%", "")
        text = text.replace(",", "")
        return text
    
    def _extract_num_from_str(self, answer: str):
        """
        Function to extract a number from a string answer

        Args:
            answer (str): answer to parse

        Returns:
            num (None, float, int): the parsed number
        """
        s = self._clean_num(answer)
        r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
        groups = re.findall(r_num, s)
        if len(groups) == 0:
            return None
        num = groups[-1][0]
        if num == '':
            return None
        if '.' in num:
            return float(num)
        return int(num)
    
    def _equal_numbers(self, prediction: str, label: str, log: bool):
        """
        Function to compare two numbers represented as strings

        Args:
            prediction (str): predicted number string
            label (str): ground truth number string
            log (bool): whether to print log statements
        """
        if not prediction or 'None' in prediction:
            return False
        else:
            try:
                if isclose(abs(round(float(label), 3)), abs(round(float(prediction), 3)), rel_tol=0.02):
                    return True
                if round(abs(float(label)), 1) == round(abs(float(prediction)), 1):
                    return True
            except Exception as e:
                if log:
                    print(f"ERROR processing label or prediction: {label} {prediction} {e}")
            return False
    
    def parse_output_by_separator(self, output_response: str, log: bool):
        """
        Function to parse LLM output for a problem, splitting the response by separator
        
        Args:
            output_response (str): the LLM response to be processed
            log (bool): whether to return log statements or not

        Returns:
            output_dict (Dict[str,str]): for each separator the text that the response had
            coherent_response (bool): whether the response contained text for all of the separators (indicates instruction following errors)
            started_response (bool): whether there was any response for the problem (indicates context length errors)
        """
        leftover_response = output_response
        output_dict = {separator: "" for separator in self.separators}
        if output_response == "ERROR: 'choices'":
            return output_dict, False, False

        coherent_response = True
        for i, separator in enumerate(self.separators):
            if separator not in leftover_response:
                if separator not in output_response: # if separator is not found at all
                    if log:
                        print(f"WARNING {separator} not found, incoherent output.")
                    coherent_response = False
                    continue
                else:
                    if log:
                        print(f"WARNING {separator} not in correct order, incoherent output.")
                    output_dict[separator] = output_response.split(separator)[1].strip()
                    coherent_response = False
                    continue
            
            # We know the separator is found, now split by it
            splitted_leftover_response = leftover_response.split(separator) # results in a list where list[1] will contain the text directly after the separator
            try: # try taking the response at index 1
                leftover_response = splitted_leftover_response[1]
                if i >= len(self.separators)-1: # if this is the last separator, the text does not need to be split further
                    output_dict[separator] = leftover_response.strip()
                else: # there is another separator after the current one
                    next_splitted = leftover_response.split(self.separators[i+1]) # split by that next separator
                    if len(next_splitted) <= 1: # if that separator doesn't exist, we can use the original unsplitted text
                        output_dict[separator] = leftover_response.strip()
                    else: # if it does exist, the list's length is greater than 1 so we take the list[0]th text as the text between the two separators
                        output_dict[separator] = next_splitted[0].strip()
            except KeyError as e:
                if log:
                    print(f"ERROR parsing {separator}, incoherent output.")
                coherent_response = False

        return output_dict, coherent_response, True
    
    def parse_output_label(self, label: str, log:bool):
        """
        Function to parse LLM output for a label, mapping to defined label_map
        
        Args:
            label (str): the LLM label to be processed
            log (bool): whether to return log statements or not

        Returns:
            final_label (str): final mapped version of the label
            coherent_label (bool): whether the label could be mapped (indicates instruction following errors)
        """
        # Remove trailing/beginning ()s, []s, **s, or whitespace
        # Remove common wrappers like quotes/brackets
        # Collapse internal whitespace
        m_label = re.sub(r"\s+", "", label)
        m_label = re.sub(r'^[\s\[\]\(\)\{\}\'"`*]+', "", m_label)
        m_label = re.sub(r'[\s\[\]\(\)\{\}\'"`*]+$', "", m_label)

        final_label = ""
        coherent_label = True
        if m_label in self.label_map:
            final_label = self.label_map[m_label]
        else:
            if self.task == "qa":
                # Check for numbers
                num = self._extract_num_from_str(m_label)
                if not num:
                    if log:
                        print(f"Unknown numeric label '{label}' modified to '{num}'.")
                    coherent_label = False
                final_label = self._clean_num(str(num)) # back to a string
            else:
                if log:
                    print(f"Unknown label '{label}' modified to '{m_label}'.")
                coherent_label = False

        return final_label, coherent_label
    
    def parse_output(self, output_json: Dict[str, Any], log: bool):
        """
        Function to parse LLM output, returning final processed labels, justifications, and invalidity flags
        
        Args:
            output_json (Dict[str,Any]): the LLM output to be processed
            log (bool): whether to return log statements or not

        Returns:
            final_output_dicts (List[Dict]): final output dicts for each output
            n_invalid (int): how many responses were unprocessable due to some issue
            n_not_started (int): how many responses were unprocessasble due to the LLM not processing in the first place: context length issues
            n_incoherent_separators (int): number of responses that were unprocessable due to the separators not being used properly
            n_incoherent_labels (int): number of responses that were unprocessable due to erroneous labels not matching label_map
        """
        # First, split each response by defined separators, keeping track of errors
        output_dicts = list()
        n_incoherent_separators = 0 # number of responses that were unprocessable due to erroneous separators in the response
        n_not_started = 0 # the number of resopnses that were unprocessable due to the LLM never returning a response (context length errors)
        for i in range(len(output_json['outputs'])):
            output_dict, coherent_response, started_response = self.parse_output_by_separator(output_json['outputs'][i]['response'], log=log)
            
            if not started_response: # if the reponse was never started
                n_not_started += 1

            if started_response and not coherent_response: # if the response was started but not coherent
                n_incoherent_separators += 1

            output_dict['coherent_separators'] = coherent_response
            output_dict['started_response'] = started_response
            output_dicts.append(output_dict)

        # Next, process the label of each response, keeping track of label errors
        n_incoherent_labels = 0 # number of responses that were unprocessable due to erroneous labels in the response (outside of separator/starting errors)
        for output_dict in output_dicts:
            label = output_dict[self.separators[0]] # depends on the task
            final_label, coherent_label = self.parse_output_label(label, log=log)

            if output_dict['started_response'] and not coherent_label and output_dict['coherent_separators']: # if the response was started, the separators were coherent, but the label was incoherent (indicates errors with label processing specifically)
                n_incoherent_labels += 1

            output_dict['label'] = final_label
            output_dict['coherent_label'] = coherent_label

        # Finally, finalize the invalid responses and create the final processed output dictionary
        n_invalid = 0 # the number of responses that were unprocessable for some reason or another
        final_output_dicts = list()
        for output_dict in output_dicts:
            final_output_dict = {'label': output_dict['label'], 'justification': output_dict[self.separators[1]]}
            if output_dict['coherent_separators'] and output_dict['coherent_label'] and output_dict['started_response']:
                final_output_dict['valid_response'] = True
            else:
                final_output_dict['valid_response'] = False
                n_invalid += 1
            final_output_dicts.append(final_output_dict)
        
        return final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels
    
    def validate_classification(self, input_json: List[Dict[str, Any]], output_dicts: List[Dict[str, Any]], log: bool):
        """
        Validation for classification tasks such as numerical fact-checking.

        Args:
            input_json (List[Dict[str, Any]]): input that contains the ground truth answer and taxonomy label
            output_dicts (List[Dict[str, Any]]): output dictionaries that contain the predicted answer and invalidity statistics

        Returns:
            n (int): number of responses used to calculate outcomes
            cf_matrix (List[List]): confusion matrix of label distribution
            accuracy (float): accuracy of responses
            weighted_f1 (float): weighted f1 score of responses
            macro_f1 (float): macro f1 score of responses
            per_class_f1s (Dict[str, float]): per class f1 score of responses
            per_taxonomy_f1s (Dict[str[Dict[str, float]]]): per taxonomy label weighted/macro f1 scores
        """
        labels = list()
        predictions = list()
        for output_dict, input_dict in zip(output_dicts, input_json):
            if output_dict['valid_response']: # include only the valid outputs
                labels.append(input_dict['label'])
                predictions.append(output_dict['label'])

        if log:
            print(f"Excluded {len(input_json) - len(labels)} invalid responses. Using {len(labels)} valid responses.\n")

            print(f"Classification Report:")
            print(classification_report(labels, predictions, labels = self.labels, target_names = self.labels, digits=4))

        cf_matrix = confusion_matrix(labels, predictions, labels=self.labels)

        accuracy = accuracy_score(labels, predictions)

        weighted_f1 = f1_score(labels, predictions, average='weighted')
        macro_f1 = f1_score(labels, predictions, average='macro')
        per_class_f1s = dict(zip(self.labels, f1_score(labels, predictions, labels=self.labels, average=None).tolist()))

        # Compute per taxonomy_label f1s

        taxonomy_labels = [input_dict['taxonomy_label'] for input_dict in input_json]
        per_taxonomy_f1s = {}
        for taxonomy_label in self.taxonomy_label_set:
            labels_t = [label for tax, label in zip(taxonomy_labels, labels) if tax == taxonomy_label]
            predictions_t = [pred for tax, pred in zip(taxonomy_labels, predictions) if tax == taxonomy_label]

            weighted_f1_t = f1_score(labels_t, predictions_t, average='weighted')
            macro_f1_t = f1_score(labels_t, predictions_t, average='macro')

            per_taxonomy_f1s[taxonomy_label] = {'weighted_f1': weighted_f1_t, 'macro_f1': macro_f1_t}

        if log:
            print(f"Confusion Matrix:\n{cf_matrix}\n")
            print(f"Accuracy: {accuracy}\n")
            print(f"Weighted-F1: {weighted_f1}")
            print(f"Macro-F1: {macro_f1}\n")
            print(f"Per-Class F1s: {per_class_f1s}")
            print(f"Per-Taxonomy F1s: {per_taxonomy_f1s}")

        return len(labels), cf_matrix.tolist(), accuracy, weighted_f1, macro_f1, per_class_f1s, per_taxonomy_f1s
    
    def validate_exact_match(self, input_json: List[Dict[str, Any]], output_dicts: List[Dict[str, Any]], log: bool):
        """
        Validation for exact match tasks such as numerical QA.

        Args:
            input_json (List[Dict[str, Any]]): input that contains the ground truth answer
            output_dicts (List[Dict[str, Any]]): output dictionaries that contain the predicted answer and invalidity statistics

        Returns:
            n (int): number of responses used to calculate outcomes
            em_accuracy (float): accuracy of responses
            n_invalid_ground_truth (int): number of ground truth labels that don't exist
        """
        n_invalid_ground_truth = 0
        n_valid = 0
        matches = 0
        mismatches = 0
        for output_dict, input_dict in zip(output_dicts, input_json):
            if output_dict['valid_response']: # include only the valid outputs
                n_valid += 1
                label = self._clean_num(input_dict['answer'].lower())
                if not label:
                    n_invalid_ground_truth += 1
                    continue
                prediction = output_dict['label'].lower()

                if 'yes' in label or 'no' in label:
                    if label == prediction: # predicted yes/no correctly
                        matches += 1
                    else: # predicted yes/no incorrectly
                        mismatches += 1
                else:
                    if self._equal_numbers(prediction, label, log=log): # numbers are equal (or close enough)
                        matches += 1
                    else: # numbers are unequal
                        mismatches += 1

        em_accuracy = matches / (mismatches + matches)

        if log:
            print(f"Excluded {len(input_json) - n_valid} invalid responses. Using {n_valid} valid responses.\n")

            print(f"Exact match accuracy: {em_accuracy}")

        return n_valid, em_accuracy, n_invalid_ground_truth
    
    def parse_judge_output(self, output_json: Dict[str, Any], log: bool):
        """
        Validation for LLM-as-a-judge output detecting numerical failures.

        Args:
            output_json (Dict[str,Any]): the LLM output to be processed with judge outcomes
            log (bool): whether to return log statements or not

        Returns:
            n_invalid (int): number of judge outcomes that were unparsable
            n_correct (int): number of parsable judge outcomes that said predicted/gt labels matched
            n_incorrect (int): number of parsable judge outcomes that said predicted/gt labels did not match
            total (Dict[str, int]): number of total valid judge outcomes with detected n_numerical_scoping, n_arithmetic, and n_looping failures
            incorrect (Dict[str, int]): number of incorrect valid judge outcomes with detected n_numerical_scoping, n_arithmetic, and n_looping failures
            caused (Dict[str, int]): number of valid judge outcomes with detected n_numerical_scoping, n_arithmetic, and n_looping failures that caused the incorrectness
        """
        # Parse into jsons
        n_invalid = 0
        for o in output_json['judge']:
            valid_json = True
            response = o['response']
            o['json'] = {}

            try:
                p_json = response[response.index('{'): response.rfind('}')+1]
                j = json.loads(p_json)
                o['json'] = j
            except Exception as e:
                valid_json = False
                n_invalid += 1
                if log:
                    print(f"ERROR parsing json {e} with json {response}")
            
            o['valid_json'] = valid_json

        # Get number with reasoning failures
        n_correct = 0
        n_incorrect = 0
        total = {"n_numerical_scoping": 0, "n_arithmetic": 0, "n_looping": 0}
        incorrect = {"n_numerical_scoping": 0, "n_arithmetic": 0, "n_looping": 0}
        caused = {"n_numerical_scoping": 0, "n_arithmetic": 0, "n_looping": 0}
        for o in output_json['judge']:
            if not o['valid_json']: # skip invalid records
                n_invalid += 1
                continue
            j = o['json']
            try:
                if j['failures']['numerical_scoping_error']['exists'].lower() == "yes":
                    total['n_numerical_scoping'] += 1
                if j['failures']['arithmetic_error']['exists'].lower() == "yes":
                    total['n_arithmetic'] += 1
                if j['failures']['infinite_loop']['exists'].lower() == "yes":
                    total['n_looping'] += 1

                if j['verdict_correct']:
                    n_correct += 1
                else:
                    n_incorrect += 1
                    if j['failures']['numerical_scoping_error']['exists'].lower() == "yes":
                        incorrect['n_numerical_scoping'] += 1
                    if j['failures']['arithmetic_error']['exists'].lower() == "yes":
                        incorrect['n_arithmetic'] += 1
                    if j['failures']['infinite_loop']['exists'].lower() == "yes":
                        incorrect['n_looping'] += 1

                    if j['failures']['numerical_scoping_error']['exists'].lower() == "yes" and j['failures']['numerical_scoping_error']['led_to_incorrect_answer'].lower() == "yes":
                        caused['n_numerical_scoping'] += 1
                    if j['failures']['arithmetic_error']['exists'].lower() == "yes" and j['failures']['arithmetic_error']['led_to_incorrect_answer'].lower() == "yes":
                        caused['n_arithmetic'] += 1
                    if j['failures']['infinite_loop']['exists'].lower() == "yes" and j['failures']['infinite_loop']['led_to_incorrect_answer'].lower() == "yes":
                        caused['n_looping'] += 1
            except Exception as e:
                o['valid_json'] = False # mark json as invalid
                n_invalid += 1
                if log:
                    print(f"ERROR with parsing {e} with json {j}")
        n_invalid, n_correct, n_incorrect, total, incorrect, caused