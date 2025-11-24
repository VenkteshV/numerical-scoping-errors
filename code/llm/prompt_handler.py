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

    def get_system_message(self) -> str:
        """Gets the system message prompt."""
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
        return system_message
    
    def get_user_message(self, claim: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]]) -> str:
        """Gets the user message prompt."""
        user_message = f"""Here are examples of fact-checking:
        {self.get_input_examples(fc_example_list)}

        Similar to the given examples, fact check the following claim using the evidence. Pay additional attention to numerical spans in claims and evidence and fact check by thinking step by step and output the label by performing entailment. Classify the entire claim strictly into one of the following categories: SUPPORTS, REFUTES, or CONFLICTING. 
        [Claim]: \"{claim}\"
        [Evidences]: {self.get_evidences(evidence_list)}

        Give the reply in the following format:
        [Label]: (SUPPORTS, REFUTES, or CONFLICTING)
        [Justification]:
        """
        return user_message
    
    def get_evidences(self, evidence_list: List[str]) -> str:
        """Gets the string version of a list of evidences."""
        evidences = ""
        for idx, evidence in enumerate(evidence_list):
            evidence += f"{idx+1}. \"{evidence}\"\n"
        return evidences
    
    def get_input_examples(self, fc_example_list: List[Dict[str, Any]]) -> str:
        """Gets the string version of a list of fact checking examples."""
        input_examples = ""
        for idx, fc_example in enumerate(fc_example_list):
            input_examples += f"{idx+1}. [Claim]: \"{fc_example['claim']}\"\n[Evidences]: {self.get_evidences(fc_example['evidence_list'])}\n [Label]: {fc_example['label']}\n\n"
        return input_examples