import re
import json
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from validation.validation_handler import ValidationHandler

class TestTimeScalingValidationHandler:
    """
    Class that does different types of test-time scaling validation for outputs.
    
    Internal States:
    - task: "fc" or "qa"
    """
    def __init__(
            self,
            task: str = "fc"):

        self.task = task
        self.validation_handler = ValidationHandler(task=task)
        self.alpha = 0.18 # the weight of the scoping score compared to the correctness score in multi-task best-of-N
        self.threshold = 0.4 # if score is above this threshold, hybrid pipeline has found a scoping error
        # if task == "fc":
        #     self.labels = ['True', 'False', 'Conflicting']
        #     self.label_map = {"SUPPORTS": "True", "REFUTES": "False", "CONFLICTING": "Conflicting"}
        #     self.separators = ["[Label]:", "[Justification]:"]
        #     self.taxonomy_label_set = ['statistical', 'temporal', 'interval', 'comparison']
        # else:
        #     self.labels = ["yes", "no"]
        #     self.label_map = {"YES": "yes", "NO": "no"}
        #     self.separators = ["[Final Answer]:", "[Justification]:"]
        #     self.taxonomy_label_set = []

    def _make_k_one_file(self, tts_dir: str, files: List[str], input_json_file: str, evidence_split: str):
        """
        Chooses random traces from the given temperature files

        Args:
            tts_dir (str): Path to base directory for test-time scaling results
            files (List[str]): Names of files to uniformly randomly take results from
            input_json_file (str): input json file results were created from
            evidence_split (str): evidence split results were created from
        """

        new_file = {'input_json_file': input_json_file,
            'outputs': [],
            'evidence_split': evidence_split,
            'advanced_judge': [],
            'verifier': {}
           }
        
        all_judges = list()
        all_scores = dict()
        all_outputs = list()
        for file in files:
            with open(f"{tts_dir}/{file}") as f:
                j = json.load(f)
            all_judges.append(j['advanced_judge'])
            for key in j['verifier']:
                if key not in all_scores:
                    all_scores[key] = list()
                all_scores[key].append(j['verifier'][key])
                    
            all_outputs.append(j['outputs'])

        random.seed(56)

        li = [random.randint(0, len(files)) for _ in range(len(all_judges[0]))]

        for i, n in enumerate(li):
            new_file['outputs'].append(all_outputs[n][i])
            new_file['advanced_judge'].append(all_judges[n][i])
            for key in all_scores:
                if key not in new_file['verifier']:
                    new_file['verifier'][key] = list()
                new_file['verifier'][key].append(all_scores[key][n][i])

        with open(f"{tts_dir}/0.json", "w") as f:
            json.dump(new_file, f)

    def _canonicalize_answer(self, answer, decimals=2):
        """
        Map raw QA answer into comparable buckets for diversity analysis.
        Returns a canonical string bucket.
        """
        if answer is None:
            return None

        s = str(answer).strip().lower()

        # yes/no normalization
        if s in {"yes", "true"}:
            return "yes"
        if s in {"no", "false"}:
            return "no"

        # try numeric parsing
        s_num = s.replace(",", "")
        match = re.fullmatch(r"[-+]?\d*\.?\d+", s_num)
        if match:
            x = float(s_num)
            return f"num:{round(x, decimals)}"

        # fallback: normalized text
        return f"text:{s}"

    def _infer_decimals_from_ground_truth(self, answer):
        s = str(answer).strip().lower().replace(",", "")
        if "." in s:
            return len(s.split(".")[-1])
        return 0

    def perform_tts_validation_fc(self, tts_dir: str, k_dict: Dict[str, List[str]], input_json_file: str, evidence_split: str, general_verifier_path: str, scoping_verifer_path: str, multitask_verifier_path: str):
        """
        Performs test-time scaling validation for different K values (numbers of candidate traces) of fact-checking traces

        Args:
            tts_dir (str): Path to base directory for test-time scaling results
            k_dict (Dict[str, List[str]]): Dictionary of k values to lists of files to take results from
                Example: {
                    '1': ["0.json"],
                    '4': ["temp_0.2_0.json", "temp_0.3_0.json", "temp_0.4_0.json", "temp_0.5_0.json"],
                    '8': ["temp_0.2_0.json", "temp_0.3_0.json", "temp_0.4_0.json", "temp_0.5_0.json",
                        "temp_0.2_1.json", "temp_0.3_1.json", "temp_0.4_1.json", "temp_0.5_1.json"],
                    '12': ["temp_0.2_0.json", "temp_0.3_0.json", "temp_0.4_0.json", "temp_0.5_0.json",
                            "temp_0.2_1.json", "temp_0.3_1.json", "temp_0.4_1.json", "temp_0.5_1.json",
                            "temp_0.2_2.json", "temp_0.3_2.json", "temp_0.4_2.json", "temp_0.5_2.json"]
                }
            input_json_file (str): input json file results were created from
            evidence_split (str): evidence split results were created from
            general_verifier_path (str): path to general correctness verifier already used to get verifier results
            scoping_verifer_path (str): path to scoping verifier already used to get verifier results
            multitask_verifier_path (str): path to multi-task scoping/correctness verifier already used to get multi-task verifier results

        Returns:
            results (Dict[str, Any]): contains the actual test-time scaling results of the different mitigation variants
            diversity_results (Dict[str, Any]): contains the diversity of the different candidate answers
        """

        # Ensure the random k file is made
        self._make_k_one_file(tts_dir=tts_dir, files=k_dict['4'], input_json_file=input_json_file, evidence_split=evidence_split)

        with open(input_json_file) as f:
            input_json = json.load(f)

        results = {}
        diversity_results = {}
        for key in k_dict:
            files = k_dict[key]
            all_output_jsons = list()
            all_output_dicts = list()
            all_advanced_judges = list()
            all_scores = list()
            all_scoping_scores = list()
            all_multitask_scores = list()
            for file in files:
                with open(f"{tts_dir}/{file}") as f:
                    output_json = json.load(f)
                final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output_json, log=False)
                n_valid, cf_matrix, accuracy, weighted_f1, macro_f1, per_class_f1s, per_taxonomy_f1s, final_output_dicts = self.validation_handler.validate_classification(input_json, final_output_dicts, log=False)
                all_output_jsons.append(output_json['outputs'])
                all_output_dicts.append(final_output_dicts)
                all_advanced_judges.append(output_json['advanced_judge'])
                all_scores.append(output_json['verifier'][general_verifier_path])
                all_scoping_scores.append(output_json['verifier'][scoping_verifer_path])
                multi_task_scores = [{'combined': score_dict['score_thinking'] - self.alpha * score_dict['scoping_score_thinking'], 'correctness': score_dict['score_thinking'], 'scoping': score_dict['scoping_score_thinking']} for score_dict in output_json['verifier'][multitask_verifier_path]]
                all_multitask_scores.append(multi_task_scores)
                assert len(final_output_dicts) == len(output_json['advanced_judge'])

            valid_labels = list()
            valid_no_error_labels = list()
            oracle_predictions = list()
            consistency_predictions = list()
            verifier_predictions = list()
            verifier_oracle_predictions = list()
            best_of_n_predictions_thinking = list()
            best_of_n_predictions_scoping_thinking = list()
            best_of_n_predictions_scoping_scores = list()
            best_of_n_predictions_multitask = list()
            best_of_n_predictions_multitask_hard = list()
            
            n_valid_answers = list()
            n_valid_no_error_answers = list()

            # outcome diversity across traces for each query
            answer_entropy_normalized = list()
            answer_disagreement = list()
            answer_n_unique = list()
            no_error_answer_entropy_normalized = list()
            no_error_answer_disagreement = list()
            no_error_answer_n_unique = list()
            
            for i, input_dict in enumerate(input_json):
                oracle_answer = False # for oracle upper bound
                valid_answers = dict() # for majority voting
                valid_no_error_answers = dict() # for reward guided majority voting
                oracle_no_error_answer = False # for oracle reward guided
                best_of_n_thinking = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_scoping_thinking = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_scoping_scores = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_multitask = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_multitask_hard = {"best_score": -1, "best_answer": None, "best_trace": None}
                
                n_answers = 0 # number of valid answers
                n_no_error_answers = 0 # number of valid reward guided answers
                for outputs, output_dict, advanced_judges, scores, scoping_scores, multitask_scores in zip(all_output_jsons, all_output_dicts, all_advanced_judges, all_scores, all_scoping_scores, all_multitask_scores):
                    if 'correct' in output_dict[i]:
                        if output_dict[i]['correct']:
                            oracle_answer = True
                        
                        if output_dict[i]['label'] not in valid_answers: # count number of answers
                            valid_answers[output_dict[i]['label']] = 1
                            n_answers += 1
                        else:
                            valid_answers[output_dict[i]['label']] += 1
                            n_answers += 1

                        # compare best scores
                        if scores[i]['score_thinking'] > best_of_n_thinking['best_score']:
                            best_of_n_thinking['best_score'] = scores[i]['score_thinking']
                            best_of_n_thinking['best_answer'] = output_dict[i]['label']
                            best_of_n_thinking['best_trace'] = outputs[i]['response']
                        if scoping_scores[i]['score_thinking'] <= self.threshold and scores[i]['score_thinking'] > best_of_n_scoping_scores['best_score']:
                            best_of_n_scoping_scores['best_score'] = scores[i]['score_thinking']
                            best_of_n_scoping_scores['best_answer'] = output_dict[i]['label']
                            best_of_n_scoping_scores['best_trace'] = outputs[i]['response']
                        if multitask_scores[i]['combined'] > best_of_n_multitask['best_score']:
                            best_of_n_multitask['best_score'] = multitask_scores[i]['combined']
                            best_of_n_multitask['best_answer'] = output_dict[i]['label']
                            best_of_n_multitask['best_trace'] = outputs[i]['response']
                        if multitask_scores[i]['scoping'] <= self.threshold and multitask_scores[i]['correctness'] > best_of_n_multitask_hard['best_score']:
                            best_of_n_multitask_hard['best_score'] = multitask_scores[i]['correctness']
                            best_of_n_multitask_hard['best_answer'] = output_dict[i]['label']
                            best_of_n_multitask_hard['best_trace'] = outputs[i]['response']

                    if 'correct' in output_dict[i] and advanced_judges[i]['valid'] and not advanced_judges[i]['filtered_scoping_error']:
                        if output_dict[i]['correct']:
                            oracle_no_error_answer = True
                            
                        if output_dict[i]['label'] not in valid_no_error_answers: # count number of answers
                            valid_no_error_answers[output_dict[i]['label']] = 1
                            n_no_error_answers += 1
                        else:
                            valid_no_error_answers[output_dict[i]['label']] += 1
                            n_no_error_answers += 1

                        # compare best scores
                        if scores[i]['score_thinking'] > best_of_n_scoping_thinking['best_score']:
                            best_of_n_scoping_thinking['best_score'] = scores[i]['score_thinking']
                            best_of_n_scoping_thinking['best_answer'] = output_dict[i]['label']
                            best_of_n_scoping_thinking['best_trace'] = outputs[i]['response']

                # Append results
                n_valid_answers.append(n_answers)
                n_valid_no_error_answers.append(n_no_error_answers)
                if n_answers:
                    valid_labels.append(input_dict['label'])
                    
                    if oracle_answer:
                        oracle_predictions.append(input_dict['label'])
                    else:
                        oracle_predictions.append(max(valid_answers, key=valid_answers.get))
            
                    consistency_predictions.append(max(valid_answers, key=valid_answers.get))

                    best_of_n_predictions_thinking.append(best_of_n_thinking['best_answer'])

                    best_of_n_predictions_scoping_scores.append(best_of_n_scoping_scores['best_answer'] if best_of_n_scoping_scores['best_answer'] else best_of_n_thinking['best_answer'])
                    best_of_n_predictions_multitask.append(best_of_n_multitask['best_answer'])
                    best_of_n_predictions_multitask_hard.append(best_of_n_multitask_hard['best_answer'] if best_of_n_multitask_hard['best_answer'] else best_of_n_thinking['best_answer'])

                    # Measure diversity
                    probs = [count / n_answers for count in valid_answers.values()]
                    entropy = -sum(p * math.log(p) for p in probs if p > 0)
                    max_entropy = math.log(3)  # fixed label space: True / False / Conflicting

                    answer_entropy_normalized.append(entropy / max_entropy if max_entropy > 0 else 0.0)
                    answer_disagreement.append(1.0 - (max(valid_answers.values()) / n_answers))
                    answer_n_unique.append(len(valid_answers))
                
                if n_no_error_answers:
                    valid_no_error_labels.append(input_dict['label'])
                    verifier_predictions.append(max(valid_no_error_answers, key=valid_no_error_answers.get))
                    if oracle_no_error_answer:
                        verifier_oracle_predictions.append(input_dict['label'])
                    else:
                        verifier_oracle_predictions.append(max(valid_no_error_answers, key=valid_no_error_answers.get))

                    best_of_n_predictions_scoping_thinking.append(best_of_n_scoping_thinking['best_answer'])
                    
                    # Measure diversity
                    probs_no_error = [count / n_no_error_answers for count in valid_no_error_answers.values()]
                    entropy_no_error = -sum(p * math.log(p) for p in probs_no_error if p > 0)
                    max_entropy_no_error = math.log(3)  # fixed label space: True / False / Conflicting

                    no_error_answer_entropy_normalized.append(entropy_no_error / max_entropy_no_error if max_entropy_no_error > 0 else 0.0)
                    no_error_answer_disagreement.append(1.0 - (max(valid_no_error_answers.values()) / n_no_error_answers))
                    no_error_answer_n_unique.append(len(valid_no_error_answers))

            # Compute f1s across different TTS aggregation types
            # Oracle
            oracle_accuracy = accuracy_score(valid_labels, oracle_predictions)
            oracle_weighted_f1 = f1_score(valid_labels, oracle_predictions, average='weighted')
            oracle_macro_f1 = f1_score(valid_labels, oracle_predictions, average='macro')
            oracle_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_labels, oracle_predictions, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Consistency
            consistency_accuracy = accuracy_score(valid_labels, consistency_predictions)
            consistency_weighted_f1 = f1_score(valid_labels, consistency_predictions, average='weighted')
            consistency_macro_f1 = f1_score(valid_labels, consistency_predictions, average='macro')
            consistency_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_labels, consistency_predictions, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Verifier Consistency
            verifier_accuracy = accuracy_score(valid_no_error_labels, verifier_predictions)
            verifier_weighted_f1 = f1_score(valid_no_error_labels, verifier_predictions, average='weighted')
            verifier_macro_f1 = f1_score(valid_no_error_labels, verifier_predictions, average='macro')
            verifier_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_no_error_labels, verifier_predictions, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Verifier Oracle
            verifier_oracle_accuracy = accuracy_score(valid_no_error_labels, verifier_oracle_predictions)
            verifier_oracle_weighted_f1 = f1_score(valid_no_error_labels, verifier_oracle_predictions, average='weighted')
            verifier_oracle_macro_f1 = f1_score(valid_no_error_labels, verifier_oracle_predictions, average='macro')
            verifier_oracle_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_no_error_labels, verifier_oracle_predictions, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Best-of-N Thinking
            best_of_n_thinking_accuracy = accuracy_score(valid_labels, best_of_n_predictions_thinking)
            best_of_n_thinking_weighted_f1 = f1_score(valid_labels, best_of_n_predictions_thinking, average='weighted')
            best_of_n_thinking_macro_f1 = f1_score(valid_labels, best_of_n_predictions_thinking, average='macro')
            best_of_n_thinking_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_labels, best_of_n_predictions_thinking, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Best-of-N Thinking + Scoping
            best_of_n_scoping_thinking_accuracy = accuracy_score(valid_no_error_labels, best_of_n_predictions_scoping_thinking)
            best_of_n_scoping_thinking_weighted_f1 = f1_score(valid_no_error_labels, best_of_n_predictions_scoping_thinking, average='weighted')
            best_of_n_scoping_thinking_macro_f1 = f1_score(valid_no_error_labels, best_of_n_predictions_scoping_thinking, average='macro')
            best_of_n_scoping_thinking_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_no_error_labels, best_of_n_predictions_scoping_thinking, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Best-of-N Scoping Scores
            best_of_n_scoping_scores_accuracy = accuracy_score(valid_labels, best_of_n_predictions_scoping_scores)
            best_of_n_scoping_scores_weighted_f1 = f1_score(valid_labels, best_of_n_predictions_scoping_scores, average='weighted')
            best_of_n_scoping_scores_macro_f1 = f1_score(valid_labels, best_of_n_predictions_scoping_scores, average='macro')
            best_of_n_scoping_scores_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_labels, best_of_n_predictions_scoping_scores, labels=['True', 'False', 'Conflicting'], average=None).tolist()))
            
            # Best-of-N Multitask
            best_of_n_multitask_accuracy = accuracy_score(valid_labels, best_of_n_predictions_multitask)
            best_of_n_multitask_weighted_f1 = f1_score(valid_labels, best_of_n_predictions_multitask, average='weighted')
            best_of_n_multitask_macro_f1 = f1_score(valid_labels, best_of_n_predictions_multitask, average='macro')
            best_of_n_multitask_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_labels, best_of_n_predictions_multitask, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            # Best-of-N Multitask Hard Threshold
            best_of_n_multitask_hard_accuracy = accuracy_score(valid_labels, best_of_n_predictions_multitask_hard)
            best_of_n_multitask_hard_weighted_f1 = f1_score(valid_labels, best_of_n_predictions_multitask_hard, average='weighted')
            best_of_n_multitask_hard_macro_f1 = f1_score(valid_labels, best_of_n_predictions_multitask_hard, average='macro')
            best_of_n_multitask_hard_per_class_f1s = dict(zip(['True', 'False', 'Conflicting'], f1_score(valid_labels, best_of_n_predictions_multitask_hard, labels=['True', 'False', 'Conflicting'], average=None).tolist()))

            
            results[key] = {
                "Oracle": {
                    "accuracy": oracle_accuracy,
                    "weighted-f1": oracle_weighted_f1,
                    "macro-f1": oracle_macro_f1,
                    "true-f1": oracle_per_class_f1s['True'],
                    "false-f1": oracle_per_class_f1s['False'],
                    "conflicting-f1": oracle_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "HP: Scoping Filtering + Oracle": {
                    "accuracy": verifier_oracle_accuracy,
                    "weighted-f1": verifier_oracle_weighted_f1,
                    "macro-f1": verifier_oracle_macro_f1,
                    "true-f1": verifier_oracle_per_class_f1s['True'],
                    "false-f1": verifier_oracle_per_class_f1s['False'],
                    "conflicting-f1": verifier_oracle_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_no_error_labels) / len(input_json)),
                },
                "Self-Consistency": {
                    "accuracy": consistency_accuracy,
                    "weighted-f1": consistency_weighted_f1,
                    "macro-f1": consistency_macro_f1,
                    "true-f1": consistency_per_class_f1s['True'],
                    "false-f1": consistency_per_class_f1s['False'],
                    "conflicting-f1": consistency_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "HP: Scoping Filtering + Self-Consistency": {
                    "accuracy": verifier_accuracy,
                    "weighted-f1": verifier_weighted_f1,
                    "macro-f1": verifier_macro_f1,
                    "true-f1": verifier_per_class_f1s['True'],
                    "false-f1": verifier_per_class_f1s['False'],
                    "conflicting-f1": verifier_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_no_error_labels) / len(input_json)),
                },
                "C-ORM: Correctness BoN": {
                    "accuracy": best_of_n_thinking_accuracy,
                    "weighted-f1": best_of_n_thinking_weighted_f1,
                    "macro-f1": best_of_n_thinking_macro_f1,
                    "true-f1": best_of_n_thinking_per_class_f1s['True'],
                    "false-f1": best_of_n_thinking_per_class_f1s['False'],
                    "conflicting-f1": best_of_n_thinking_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "HP: Scoping Filtering + C-ORM: Correctness BoN": {
                    "accuracy": best_of_n_scoping_thinking_accuracy,
                    "weighted-f1": best_of_n_scoping_thinking_weighted_f1,
                    "macro-f1": best_of_n_scoping_thinking_macro_f1,
                    "true-f1": best_of_n_scoping_thinking_per_class_f1s['True'],
                    "false-f1": best_of_n_scoping_thinking_per_class_f1s['False'],
                    "conflicting-f1": best_of_n_scoping_thinking_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_no_error_labels) / len(input_json)),
                },
                "S-ORM: Scoping Filtering + C-ORM: Correctness BoN": {
                    "accuracy": best_of_n_scoping_scores_accuracy,
                    "weighted-f1": best_of_n_scoping_scores_weighted_f1,
                    "macro-f1": best_of_n_scoping_scores_macro_f1,
                    "true-f1": best_of_n_scoping_scores_per_class_f1s['True'],
                    "false-f1": best_of_n_scoping_scores_per_class_f1s['False'],
                    "conflicting-f1": best_of_n_scoping_scores_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "MT-ORM: Scoping Filtering + MT-ORM: Correctness BoN": {
                    "accuracy": best_of_n_multitask_hard_accuracy,
                    "weighted-f1": best_of_n_multitask_hard_weighted_f1,
                    "macro-f1": best_of_n_multitask_hard_macro_f1,
                    "true-f1": best_of_n_multitask_hard_per_class_f1s['True'],
                    "false-f1": best_of_n_multitask_hard_per_class_f1s['False'],
                    "conflicting-f1": best_of_n_multitask_hard_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "MT-ORM: Scoping & Correctness BoN": {
                    "accuracy": best_of_n_multitask_accuracy,
                    "weighted-f1": best_of_n_multitask_weighted_f1,
                    "macro-f1": best_of_n_multitask_macro_f1,
                    "true-f1": best_of_n_multitask_per_class_f1s['True'],
                    "false-f1": best_of_n_multitask_per_class_f1s['False'],
                    "conflicting-f1": best_of_n_multitask_per_class_f1s['Conflicting'],
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
            }

            # Diversity
            mean_answer_entropy_normalized = sum(answer_entropy_normalized) / len(answer_entropy_normalized) if answer_entropy_normalized else 0.0
            mean_answer_disagreement = sum(answer_disagreement) / len(answer_disagreement) if answer_disagreement else 0.0
            mean_answer_n_unique = sum(answer_n_unique) / len(answer_n_unique) if answer_n_unique else 0.0

            mean_no_error_answer_entropy_normalized = sum(no_error_answer_entropy_normalized) / len(no_error_answer_entropy_normalized) if no_error_answer_entropy_normalized else 0.0
            mean_no_error_answer_disagreement = sum(no_error_answer_disagreement) / len(no_error_answer_disagreement) if no_error_answer_disagreement else 0.0
            mean_no_error_answer_n_unique = sum(no_error_answer_n_unique) / len(no_error_answer_n_unique) if no_error_answer_n_unique else 0.0

            diversity_results[key] = {
                "mean-answer-entropy-normalized": mean_answer_entropy_normalized,
                "verifier-mean-answer-entropy-normalized": mean_no_error_answer_entropy_normalized,
                "mean-answer-disagreement": mean_answer_disagreement,
                "verifier-mean-answer-disagreement": mean_no_error_answer_disagreement,
                "mean-n-unique-answers": mean_answer_n_unique,
                "verifier-mean-n-unique-answers": mean_no_error_answer_n_unique,
            }

        return results, diversity_results
    
    def perform_tts_validation_qa(self, tts_dir: str, k_dict: Dict[str, List[str]], input_json_file: str, evidence_split: str, general_verifier_path: str, scoping_verifer_path: str, multitask_verifier_path: str):
        """
        Performs test-time scaling validation for different K values (numbers of candidate traces) of question-answering traces

        Args:
            tts_dir (str): Path to base directory for test-time scaling results
            k_dict (Dict[str, List[str]]): Dictionary of k values to lists of files to take results from
                Example: {
                    '1': ["0.json"],
                    '4': ["temp_0.2_0.json", "temp_0.3_0.json", "temp_0.4_0.json", "temp_0.5_0.json"],
                    '8': ["temp_0.2_0.json", "temp_0.3_0.json", "temp_0.4_0.json", "temp_0.5_0.json",
                        "temp_0.2_1.json", "temp_0.3_1.json", "temp_0.4_1.json", "temp_0.5_1.json"],
                    '12': ["temp_0.2_0.json", "temp_0.3_0.json", "temp_0.4_0.json", "temp_0.5_0.json",
                            "temp_0.2_1.json", "temp_0.3_1.json", "temp_0.4_1.json", "temp_0.5_1.json",
                            "temp_0.2_2.json", "temp_0.3_2.json", "temp_0.4_2.json", "temp_0.5_2.json"]
                }
            input_json_file (str): input json file results were created from
            evidence_split (str): evidence split results were created from
            general_verifier_path (str): path to general correctness verifier already used to get verifier results
            scoping_verifer_path (str): path to scoping verifier already used to get verifier results
            multitask_verifier_path (str): path to multi-task scoping/correctness verifier already used to get multi-task verifier results

        Returns:
            results (Dict[str, Any]): contains the actual test-time scaling results of the different mitigation variants
            diversity_results (Dict[str, Any]): contains the diversity of the different candidate answers
        """

        # Ensure the random k file is made
        self._make_k_one_file(tts_dir=tts_dir, files=k_dict['4'], input_json_file=input_json_file, evidence_split=evidence_split)

        with open(input_json_file) as f:
            input_json = json.load(f)

        results = {}
        diversity_results = {}
        for key in k_dict:
            files = k_dict[key]
            all_output_jsons = list()
            all_output_dicts = list()
            all_advanced_judges = list()
            all_scores = list()
            all_scoping_scores = list()
            all_multitask_scores = list()
            for file in files:
                with open(f"{tts_dir}/{file}") as f:
                    output_json = json.load(f)
                final_output_dicts, n_invalid, n_not_started, n_incoherent_separators, n_incoherent_labels = self.validation_handler.parse_output(output_json, log=False)
                n_valid, em_accuracy, n_invalid_ground_truth, final_output_dicts = self.validation_handler.validate_exact_match(input_json, final_output_dicts, log=False)
                all_output_jsons.append(output_json['outputs'])
                all_output_dicts.append(final_output_dicts)
                all_advanced_judges.append(output_json['advanced_judge'])
                all_scores.append(output_json['verifier'][general_verifier_path])
                all_scoping_scores.append(output_json['verifier'][scoping_verifer_path])
                multi_task_scores = [{'combined': score_dict['score_thinking'] - self.alpha * score_dict['scoping_score_thinking'], 'correctness': score_dict['score_thinking'], 'scoping': score_dict['scoping_score_thinking']} for score_dict in output_json['verifier'][multitask_verifier_path]]
                all_multitask_scores.append(multi_task_scores)
                assert len(final_output_dicts) == len(output_json['advanced_judge'])

            valid_labels = list()
            valid_no_error_labels = list()
            oracle_predictions = list()
            consistency_predictions = list()
            verifier_predictions = list()
            verifier_oracle_predictions = list()
            best_of_n_predictions_thinking = list()
            best_of_n_predictions_scoping_thinking = list()
            best_of_n_predictions_scoping_scores = list()
            best_of_n_predictions_multitask = list()
            best_of_n_predictions_multitask_hard = list()
            
            n_valid_answers = list()
            n_valid_no_error_answers = list()

            # outcome diversity across traces for each query
            answer_entropy_normalized = list()
            answer_disagreement = list()
            answer_n_unique = list()
            no_error_answer_entropy_normalized = list()
            no_error_answer_disagreement = list()
            no_error_answer_n_unique = list()
            
            for i, input_dict in enumerate(input_json):
                decimals = self._infer_decimals_from_ground_truth(input_dict['answer'])

                oracle_answer = False # for oracle upper bound
                valid_answers = dict() # for majority voting
                valid_no_error_answers = dict() # for reward guided majority voting
                oracle_no_error_answer = False # for oracle reward guided
                best_of_n_thinking = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_scoping_thinking = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_scoping_scores = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_multitask = {"best_score": -1, "best_answer": None, "best_trace": None}
                best_of_n_multitask_hard = {"best_score": -1, "best_answer": None, "best_trace": None}
                
                n_answers = 0 # number of valid answers
                n_no_error_answers = 0 # number of valid reward guided answers
                correctness_map = dict()
                valid_answer_buckets = dict() # for diversity over canonicalized answers
                valid_no_error_answer_buckets = dict() # for diversity over canonicalized filtered answers
                for outputs, output_dict, advanced_judges, scores, scoping_scores, multitask_scores in zip(all_output_jsons, all_output_dicts, all_advanced_judges, all_scores, all_scoping_scores, all_multitask_scores):
                    if 'correct' in output_dict[i]:
                        if output_dict[i]['correct']:
                            oracle_answer = True
                        
                        if output_dict[i]['label'] not in valid_answers: # count number of answers
                            valid_answers[output_dict[i]['label']] = 1
                            n_answers += 1
                            correctness_map[output_dict[i]['label']] = output_dict[i]['correct']
                        else:
                            valid_answers[output_dict[i]['label']] += 1
                            n_answers += 1
                            
                        # answer diversity
                        canonical_label = self._canonicalize_answer(output_dict[i]['label'], decimals=decimals)
                        if canonical_label not in valid_answer_buckets:
                            valid_answer_buckets[canonical_label] = 1
                        else:
                            valid_answer_buckets[canonical_label] += 1

                        # compare best scores
                        if scores[i]['score_thinking'] > best_of_n_thinking['best_score']:
                            best_of_n_thinking['best_score'] = scores[i]['score_thinking']
                            best_of_n_thinking['best_answer'] = correctness_map[output_dict[i]['label']]
                            best_of_n_thinking['best_trace'] = outputs[i]['response']
                        if scoping_scores[i]['score_thinking'] <= self.threshold and scores[i]['score_thinking'] > best_of_n_scoping_scores['best_score']:
                            best_of_n_scoping_scores['best_score'] = scores[i]['score_thinking']
                            best_of_n_scoping_scores['best_answer'] = correctness_map[output_dict[i]['label']]
                            best_of_n_scoping_scores['best_trace'] = outputs[i]['response']
                        if multitask_scores[i]['combined'] > best_of_n_multitask['best_score']:
                            best_of_n_multitask['best_score'] = multitask_scores[i]['combined']
                            best_of_n_multitask['best_answer'] = correctness_map[output_dict[i]['label']]
                            best_of_n_multitask['best_trace'] = outputs[i]['response']
                        if multitask_scores[i]['scoping'] <= self.threshold and multitask_scores[i]['correctness'] > best_of_n_multitask_hard['best_score']:
                            best_of_n_multitask_hard['best_score'] = multitask_scores[i]['correctness']
                            best_of_n_multitask_hard['best_answer'] = correctness_map[output_dict[i]['label']]
                            best_of_n_multitask_hard['best_trace'] = outputs[i]['response']

                    if 'correct' in output_dict[i] and advanced_judges[i]['valid'] and not advanced_judges[i]['filtered_scoping_error']:
                        if output_dict[i]['correct']:
                            oracle_no_error_answer = True
                            
                        if output_dict[i]['label'] not in valid_no_error_answers: # count number of answers
                            valid_no_error_answers[output_dict[i]['label']] = 1
                            n_no_error_answers += 1
                        else:
                            valid_no_error_answers[output_dict[i]['label']] += 1
                            n_no_error_answers += 1

                        # Measure diversity
                        canonical_label_no_error = self._canonicalize_answer(output_dict[i]['label'], decimals=decimals)
                        if canonical_label_no_error not in valid_no_error_answer_buckets:
                            valid_no_error_answer_buckets[canonical_label_no_error] = 1
                        else:
                            valid_no_error_answer_buckets[canonical_label_no_error] += 1

                        # compare best scores
                        if scores[i]['score_thinking'] > best_of_n_scoping_thinking['best_score']:
                            best_of_n_scoping_thinking['best_score'] = scores[i]['score_thinking']
                            best_of_n_scoping_thinking['best_answer'] = correctness_map[output_dict[i]['label']]
                            best_of_n_scoping_thinking['best_trace'] = outputs[i]['response']

                # Append results
                n_valid_answers.append(n_answers)
                n_valid_no_error_answers.append(n_no_error_answers)
                if n_answers:
                    valid_labels.append(input_dict['answer'])
                    
                    if oracle_answer:
                        oracle_predictions.append(True)
                    else:
                        oracle_predictions.append(correctness_map[max(valid_answers, key=valid_answers.get)])
            
                    consistency_predictions.append(correctness_map[max(valid_answers, key=valid_answers.get)])

                    best_of_n_predictions_thinking.append(best_of_n_thinking['best_answer'])

                    best_of_n_predictions_scoping_scores.append(best_of_n_scoping_scores['best_answer'] if best_of_n_scoping_scores['best_answer'] else best_of_n_thinking['best_answer'])
                    best_of_n_predictions_multitask.append(best_of_n_multitask['best_answer'])
                    best_of_n_predictions_multitask_hard.append(best_of_n_multitask_hard['best_answer'] if best_of_n_multitask_hard['best_answer'] else best_of_n_thinking['best_answer'])

                    # Measure diversity
                    probs = [count / n_answers for count in valid_answer_buckets.values()]
                    entropy = -sum(p * math.log(p) for p in probs if p > 0)
                    max_entropy = math.log(len(valid_answer_buckets)) if len(valid_answer_buckets) > 1 else 1.0

                    answer_entropy_normalized.append(entropy / max_entropy if len(valid_answer_buckets) > 1 else 0.0)
                    answer_disagreement.append(1.0 - (max(valid_answer_buckets.values()) / n_answers))
                    answer_n_unique.append(len(valid_answer_buckets))
                
                if n_no_error_answers:
                    valid_no_error_labels.append(input_dict['answer'])
                    best_answer = max(valid_no_error_answers, key=valid_no_error_answers.get) # pick majority vote
                    verifier_predictions.append(correctness_map[best_answer])
                    if oracle_no_error_answer:
                        verifier_oracle_predictions.append(True)
                    else:
                        verifier_oracle_predictions.append(correctness_map[best_answer])

                    best_of_n_predictions_scoping_thinking.append(best_of_n_scoping_thinking['best_answer'])
                    
                    # Measure diversity
                    probs_no_error = [count / n_no_error_answers for count in valid_no_error_answer_buckets.values()]
                    entropy_no_error = -sum(p * math.log(p) for p in probs_no_error if p > 0)
                    max_entropy_no_error = math.log(len(valid_no_error_answer_buckets)) if len(valid_no_error_answer_buckets) > 1 else 1.0

                    no_error_answer_entropy_normalized.append(entropy_no_error / max_entropy_no_error if len(valid_no_error_answer_buckets) > 1 else 0.0)
                    no_error_answer_disagreement.append(1.0 - (max(valid_no_error_answer_buckets.values()) / n_no_error_answers))
                    no_error_answer_n_unique.append(len(valid_no_error_answer_buckets))

            # Compute accuracies across different TTS aggregation types
            # Oracle
            oracle_em = sum(oracle_predictions) / len(oracle_predictions)
    
            # Consistency
            consistency_em = sum(consistency_predictions) / len(consistency_predictions)
            
            # Verifier
            verifier_em = sum(verifier_predictions) / len(verifier_predictions)
            
            # Verifier Oracle
            verifier_oracle_em = sum(verifier_oracle_predictions) / len(verifier_oracle_predictions)
            
            # Best-of-N Thinking
            best_of_n_thinking_em = sum(best_of_n_predictions_thinking) / len(best_of_n_predictions_thinking)
                        
            # Best-of-N Thinking + Scoping
            best_of_n_scoping_thinking_em = sum(best_of_n_predictions_scoping_thinking) / len(best_of_n_predictions_scoping_thinking)

            # Best-of-N Scoping Scores
            best_of_n_scoping_scores_em = sum(best_of_n_predictions_scoping_scores) / len(best_of_n_predictions_scoping_scores)

            # Best-of-N Multitask
            best_of_n_multitask_em = sum(best_of_n_predictions_multitask) / len(best_of_n_predictions_multitask)
            
            # Best-of-N Multitask Hard Threshold
            best_of_n_multitask_hard_em = sum(best_of_n_predictions_multitask_hard) / len(best_of_n_predictions_multitask_hard)
            
            results[key] = {
                "Oracle": {
                    "exact-match-accuracy": oracle_em,
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "HP: Scoping Filtering + Oracle": {
                    "exact-match-accuracy": verifier_oracle_em,
                    "invalidity-rate": 1 - (len(valid_no_error_labels) / len(input_json)),
                },
                "Self-Consistency": {
                    "exact-match-accuracy": consistency_em,
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "HP: Scoping Filtering + Self-Consistency": {
                    "exact-match-accuracy": verifier_em,
                    "invalidity-rate": 1 - (len(valid_no_error_labels) / len(input_json)),
                },
                "C-ORM: Correctness BoN": {
                    "exact-match-accuracy": best_of_n_thinking_em,
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "HP: Scoping Filtering + C-ORM: Correctness BoN": {
                    "exact-match-accuracy": best_of_n_scoping_thinking_em,
                    "invalidity-rate": 1 - (len(valid_no_error_labels) / len(input_json)),
                },
                "S-ORM: Scoping Filtering + C-ORM: Correctness BoN": {
                    "exact-match-accuracy": best_of_n_scoping_scores_em,
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "MT-ORM: Scoping Filtering + C-ORM: Correctness BoN": {
                    "exact-match-accuracy": best_of_n_multitask_hard_em,
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
                "MT-ORM: Scoping & Correctness BoN": {
                    "exact-match-accuracy": best_of_n_multitask_em,
                    "invalidity-rate": 1 - (len(valid_labels) / len(input_json)),
                },
            }

            # Diversity
            mean_answer_entropy_normalized = sum(answer_entropy_normalized) / len(answer_entropy_normalized) if answer_entropy_normalized else 0.0
            mean_answer_disagreement = sum(answer_disagreement) / len(answer_disagreement) if answer_disagreement else 0.0
            mean_answer_n_unique = sum(answer_n_unique) / len(answer_n_unique) if answer_n_unique else 0.0

            mean_no_error_answer_entropy_normalized = sum(no_error_answer_entropy_normalized) / len(no_error_answer_entropy_normalized) if no_error_answer_entropy_normalized else 0.0
            mean_no_error_answer_disagreement = sum(no_error_answer_disagreement) / len(no_error_answer_disagreement) if no_error_answer_disagreement else 0.0
            mean_no_error_answer_n_unique = sum(no_error_answer_n_unique) / len(no_error_answer_n_unique) if no_error_answer_n_unique else 0.0

            diversity_results[key] = {
                "mean-answer-entropy-normalized": mean_answer_entropy_normalized,
                "verifier-mean-answer-entropy-normalized": mean_no_error_answer_entropy_normalized,
                "mean-answer-disagreement": mean_answer_disagreement,
                "verifier-mean-answer-disagreement": mean_no_error_answer_disagreement,
                "mean-n-unique-answers": mean_answer_n_unique,
                "verifier-mean-n-unique-answers": mean_no_error_answer_n_unique,
            }

        return results, diversity_results
    
    def plot_results(self, results, plot_root="./"):
        """
        Plot a metric across different k values

        Args:
            results (Dict[str, Any]): results dictionary from tts_validation functions above
            plot_root (str): path to where plots should be saved
        """
        plt.style.use('seaborn-v0_8')
        metric_name = 'weighted-f1' if self.task == "fc" else "exact-match-accuracy"

        k_values = list(results.keys())
        strategies = list(results[k_values[0]].keys())

        x = np.arange(len(strategies))
        bar_width = 0.18

        fig, ax = plt.subplots(figsize=(8.8, 4))

        # Draw grouped bars
        for i, k_value in enumerate(k_values):
            values = [results[k_value][strategy][metric_name] for strategy in strategies]
            ax.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=f'K={k_value}',
                linewidth=0.5
            )

        # Center x-ticks
        ax.set_xticks(x + bar_width * (len(k_values) - 1) / 2)
        ax.set_xticklabels([fill(s, width=12) for s in strategies])

        ax.set_ylabel("Weighted-F1" if self.task == "fc" else "Exact Match Accuracy")
        ax.set_xlabel("Mitigation Strategy")
        ax.set_ylim(0, 1)

        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, color=(0.4176470588235294, 0.4176470588235294, 0.4490196078431372, 1.0))
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(title=None, frameon=True, shadow=True, alignment="right", bbox_to_anchor=(1.1, 1))
        ax.set_facecolor("white")
        plt.tight_layout()
        plt.savefig(f"{plot_root}{self.task}_metric_across_k_values.pdf", format="pdf")
        plt.show()

