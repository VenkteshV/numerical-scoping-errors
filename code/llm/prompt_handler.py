from typing import Dict, Any, List

class PromptHandler:
    """
    Class that creates system and user messages for prompting.
    
    Internal States:
    - task: "fc" or "qa"
    """
    def __init__(
            self,
            task: str = "fc"):

        self.task = task
        if task == "fc":
            self.label_map = {"True": "SUPPORTS", "False": "REFUTES", "Conflicting": "CONFLICTING"}
        else:
            self.label_map = {"yes": "YES", "no": "NO"}

    def get_system_message(self) -> str:
        """Gets the system message prompt."""
        if self.task == "fc":
            system_message = f"""You are a fact-checking assistant tasked with evaluating claims based on provided evidence.
            Your task:
                1. Carefully analyze the important details from both the claim and the evidence provided.
                2. Reason through the evidence step-by-step, synthesizing relevant information to assess the overall veracity of the claim.
                3. Finally, classify the claim's veracity into one of the following categories:
                    - **SUPPORTS**: The evidence fully supports the claim.
                    - **REFUTES**: The evidence directly contradicts the claim.
                    - **CONFLICTING**: The evidence contains contradictory elements or is inconclusive.
            Response Format:
            - [Label]: (SUPPORTS, REFUTES, or CONFLICTING)
            - [Justification]: The reasoning steps used that led to your classification of the claim.
            """
        else:
            system_message = f"""You are a question-answering assistant tasked with answering questions based on provided evidence.
            Your task:
                1. Carefully analyze and decompose the important details from the question.
                2. Reason through the evidence in the table and text step-by-step, looking for information to help you answer the question.
                3. Finally, output the final answer to the question in numerical or binary form without extra text.
            Response Format:
            - [Final Answer]: (YES, NO, or some number)
            - [Justification]: The reasoning steps used that led to your final answer.
            """
        return system_message
    
    def get_user_message(self, query: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]]) -> str:
        """Gets the user message prompt."""
        if self.task == "fc":
            user_message = f"""Here are examples of fact-checking:
            {self.get_input_examples(fc_example_list)}

            Similar to the given examples, fact check the following claim using the evidence. Pay additional attention to numerical spans in claims and evidence and fact check by thinking step by step and output the label by performing entailment. Classify the entire claim strictly into one of the following categories: SUPPORTS, REFUTES, or CONFLICTING. 
            [Claim]: \"{query}\"
            [Evidences]: {self.get_evidences(evidence_list)}

            Give the reply in the following format:
            [Label]: (SUPPORTS, REFUTES, or CONFLICTING)
            [Justification]:
            """
        else:
            user_message = f"""Here are examples of question answering:
            {self.get_input_examples(fc_example_list)}

            Similar to the given examples, answer the following questions using the table and text evidence. Give the step by step solution justification. Output the final answer as binary (YES or NO) or as a number without additional text.
            [Question]: \"{query}\"
            [Evidences]: {self.get_evidences(evidence_list)}

            Give the reply in the following format:
            [Final Answer]: (YES, NO, or some number)
            [Justification]:
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