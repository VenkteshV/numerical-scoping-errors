from typing import Dict, Any, List
import json

class PromptHandler:
    """
    Class that creates system and user messages for prompting.
    
    Internal States:
    - task: "fc" or "qa"
    """
    def __init__(
            self,
            task: str = "fc",
            system_prompt_file: str = "llm/system_prompts.json"):

        self.task = task
        self.system_prompt_file = system_prompt_file
        self.system_prompts = self.load_system_prompts(system_prompt_file)
        if task == "fc":
            self.label_map = {"True": "SUPPORTS", "False": "REFUTES", "Conflicting": "CONFLICTING"}
        else:
            self.label_map = {"yes": "YES", "no": "NO"}

    def load_system_prompts(self, system_prompt_file: str):
        """Loads the parameter json file."""
        with open(system_prompt_file, 'r') as f:
            return json.load(f)

    def get_system_message(self, variant: str="base") -> str:
        """
        Gets the system message prompt.
        
        Args:
            variant (str): which system prompt to retrieve: "base", "judge", "non_extractive_check" or "extractive_check"
        """
        
        return self.system_prompts[variant][self.task]
    
    def get_user_message(self, query: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]], metadata: Dict[str, Any]=None, variant: str="base") -> str:
        """
        Gets the user message prompt.
        
        Args:
            query (str): claim or question to fact-check or answer
            fc_example_list (List[Dict[str, Any]]): few-shot examples to include in base prompts
            metadata (Dict[str, Any]): metadata to be included for judge prompts
            variant (str): which system prompt to retrieve: "base", "judge", "non_extractive_check" or "extractive_check"
        """
        if self.task == "fc":
            if variant == "base":
                user_message = f"""Here are examples of fact-checking:
{self.get_input_examples(fc_example_list)}

Similar to the given examples, fact check the following claim using the evidence. Pay additional attention to numerical spans in claims and evidence and fact check by thinking step by step and output the label by performing entailment. Classify the entire claim strictly into one of the following categories: SUPPORTS, REFUTES, or CONFLICTING. 
[Claim]: \"{query}\"
[Evidences]: {self.get_evidences(evidence_list)}

Give the reply in the following format:
[Label]: (SUPPORTS, REFUTES, or CONFLICTING)
[Justification]:
"""
            elif variant == "judge":
                user_message = f"""Judge the following fact-checking output for numerical reasoning failures. Do NOT re-run the full fact-check from scratch unless needed. Detect and precisely localize scoping, arithmetic, and looping failures in the fact-checking LLM's output, only if present. Output only the JSON with no other text.
== INPUTS ==
[Claim]
{query}

[Evidence]
{self.get_evidences(evidence_list)} 

[Fact-checking LLM Output: Reasoning Trace, LLM Predicted Label, and LLM Justification]
{metadata['response']}

[Ground Truth Label (Human)]
{self.label_map[metadata['actual_label']] if metadata['actual_label'] in self.label_map else metadata['actual_label']}
== END INPUTS ==

== YOUR OUTPUT ==
"""
                
            elif variant=="non_extractive_check":
                statistics = "\n\n".join(metadata['statistics'])
                user_message = f"""For each STATISTIC (IDs 1 - {len(metadata['statistics'])}), classify STATISTIC as extrapolative vs parametric, then evaluate correctness per the system rules. Return ONLY the JSON.

QUESTION:
<QUESTION>
{query}
</QUESTION>

EVIDENCES:
<EVIDENCES>
{self.get_evidences(evidence_list)}
</EVIDENCES>

STATISTICS:
<STATISTICS>
{statistics}
</STATISTICS>

THINKING:
<THINKING>
{metadata['response']}
</THINKING>
"""    

            else:
                statistics = "\n\n".join(metadata['statistics'])
                user_message = f"""For each STATISTIC 1-{len(metadata['statistics'])}, check if CLAIM/EVIDENCE was interpreted correctly in THINKING. span_quote must be copied whole sentences exactly from CLAIM or EVIDENCE. Return only the JSON.

CLAIM:
<CLAIM>
{query}
</CLAIM>

EVIDENCES:
<EVIDENCES>
{self.get_evidences(evidence_list)}
</EVIDENCES>

THINKING:
<THINKING>
{metadata['response']}
</THINKING>

STATISTICS:
<STATISTICS>
{statistics}
</STATISTICS>
"""

        else:
            if variant == "base":
                user_message = f"""Here are examples of question answering:
{self.get_input_examples(fc_example_list)}

Similar to the given examples, answer the following questions using the table and text evidence. Give the step by step solution justification. Output the final answer as binary (YES or NO) or as a number without additional text.
[Question]: \"{query}\"
[Evidences]: {self.get_evidences(evidence_list)}

Give the reply in the following format:
[Final Answer]: (YES, NO, or some number)
[Justification]:
"""
            elif variant == "judge":
                user_message = f"""Judge the following question-answering output for numerical reasoning failures. Do NOT re-answer the question from scratch unless needed. Detect and precisely localize scoping, arithmetic, and looping failures in the question-answering LLM's output, only if present. Output only the JSON with no other text.
== INPUTS ==
[Question]
{query}

[Evidence]
{self.get_evidences(evidence_list)} 

[Question-Answering LLM Output: Reasoning Trace, LLM Predicted Answer, and LLM Justification]
{metadata['response']}

[Ground Truth Answer]
{self.label_map[metadata['actual_label']] if metadata['actual_label'] in self.label_map else metadata['actual_label']}
== END INPUTS ==

== YOUR OUTPUT ==
"""
            elif variant=="non_extractive_check":
                statistics = "\n\n".join(metadata['statistics'])
                user_message = f"""For each STATISTIC (IDs 1 - {len(metadata['statistics'])}), classify STATISTIC as extrapolative vs parametric, then evaluate correctness per the system rules. Return ONLY the JSON.

QUESTION:
<QUESTION>
{query}
</QUESTION>

EVIDENCES:
<EVIDENCES>
{self.get_evidences(evidence_list)}
</EVIDENCES>

STATISTICS:
<STATISTICS>
{statistics}
</STATISTICS>

THINKING:
<THINKING>
{metadata['response']}
</THINKING>
"""
            
            else:
                statistics = "\n\n".join(metadata['statistics'])
                user_message = f"""For each STATISTIC 1-{len(metadata['statistics'])}, check if QUESTION/EVIDENCE was interpreted correctly in THINKING. span_quote must be copied whole sentences or table rows along with header row exactly from QUESTION or EVIDENCE. Return only the JSON.

QUESTION:
<QUESTION>
{query}
</QUESTION>

EVIDENCES:
<EVIDENCES>
{self.get_evidences(evidence_list)}
</EVIDENCES>

THINKING:
<THINKING>
{metadata['response']}
</THINKING>

STATISTICS:
<STATISTICS>
{statistics}
</STATISTICS>
"""
        return user_message
    
    def get_evidences(self, evidence_list: List[str]) -> str:
        """Gets the string version of a list of evidences."""
        evidences = ""
        if self.task == "fc":
            for idx, evidence in enumerate(evidence_list):
                evidences += f"{idx+1}. \"{evidence}\"\n"
        else:
            for idx, evidence in enumerate(evidence_list):
                evidences += f"{idx+1}. \"{evidence}\"\n"
        return evidences
    
    def get_input_examples(self, fc_example_list: List[Dict[str, Any]]) -> str:
        """Gets the string version of a list of fact checking examples."""
        input_examples = ""
        if self.task == "fc":
            for idx, fc_example in enumerate(fc_example_list):
                label = fc_example['label'] if fc_example['label'] not in self.label_map else self.label_map[fc_example['label']]
                input_examples += f"{idx+1}. [Claim]: \"{fc_example['claim']}\"\n[Evidences]: {self.get_evidences(fc_example['evidence_list'])}\n[Label]: {label}\n\n"
        else:
            for idx, fc_example in enumerate(fc_example_list):
                answer = fc_example['answer'] if fc_example['answer'] not in self.label_map else self.label_map[fc_example['answer']]
                input_examples += f"{idx+1}. [Question]: \"{fc_example['question']}\"\n[Evidences]: {self.get_evidences(fc_example['evidence_list'])}\n[Final Answer]: {answer}\n\n"
        
        return input_examples