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

    def get_system_message(self, judge: bool=False) -> str:
        """Gets the system message prompt."""
        if self.task == "fc":
            if not judge:
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
                system_message = """You are an evaluation model tasked with analyzing the reasoning traces produced by a fact-checking LLM.

You will be given:
1) A claim
2) Evidence used for fact-checking
3) A reasoning trace produced by an LLM
4) The LLM's predicted label (SUPPORTS / REFUTES / CONFLICTING), if present
5) The LLM's justification, if present
6) The human-annotated ground-truth label (SUPPORTS / REFUTES / CONFLICTING)

Your task is to determine whether the reasoning trace contains any of the following failure types and, if so, whether they contributed to an incorrect final prediction.

IMPORTANT: Reasoning traces can contain no numerical reasoning failures. Do not infer failures unless there is clear evidence in the reasoning trace.

FAILURE TYPES
1) Numerical scoping error
- A numerical quantity is incorrectly bound to an entity, attribute, unit, population, or scope different from what is supported by the evidence.
- Examples: mixing up two different “rates,” attaching a number to the wrong country/year, confusing “revenue” vs “profit,” swapping numerator/denominator, applying a statistic to the wrong population, mismatching units (%, dollars, people, acres), or using evidence about one entity to justify a claim about another.

2) Arithmetic errors:
- The reasoning combines quantities incorrectly (wrong addition/subtraction/multiplication/division), miscomputes differences/percent changes, misapplies rounding, double-counts, or performs invalid aggregation.

3) Infinite loops / no final verdict:
The reasoning trace fails to terminate or does not produce a final label due to repetitive reasoning.
- If a final label is present and the trace ends cleanly, mark NO.

CAUSALITY RULE
For each failure type, you must decide whether it contributed to an incorrect final answer:
- If the predicted label matches the ground-truth label, any detected failure MUST be marked as not contributing to an incorrect answer.
- If the predicted label does not match the ground-truth label:
    - "yes" if the failure plausibly caused or materially contributed to choosing the wrong label.
    - "no" if the failure exists but is irrelevant to why the final label is wrong.
If you are unsure about causality, choose "unclear" and explain why.

YOUR OUTPUT FORMAT (STRICT)
Return a single JSON object with exactly these top-level keys:
{
"verdict_correct": boolean,                 // whether predicted label equals ground-truth label (true or false)
"failures": {
    "numerical_scoping_error": {
    "exists": "yes"|"no",
    "where": [ { "span_quote": string, "explanation": string } ],
    "led_to_incorrect_answer": "yes"|"no"|"unclear"
    },
    "arithmetic_error": {
    "exists": "yes"|"no",
    "where": [ { "span_quote": string, "explanation": string } ],
    "led_to_incorrect_answer": "yes"|"no"|"unclear"
    },
    "infinite_loop": {
    "exists": "yes"|"no",
    "where": [ { "span_quote": string, "explanation": string } ],
    "led_to_incorrect_answer": "yes"|"no"|"unclear"
    }
},
"brief_summary": string                     // 1–3 sentences, focusing on the most important detected issue(s)
}

RULES
- If "exists" is "no", "where" MUST be an empty list [].
- "span_quote" should be a short direct quote (or two) from the reasoning trace or justification that localizes the failure.
- Keep explanations concrete: name the numbers, entities, units, and what was mismatched or miscomputed.
- Do not output anything except the JSON."""
        else:
            if not judge:
                system_message = f"""You are a question-answering assistant tasked with answering questions based on provided evidence.
Your task:
    1. Carefully analyze and decompose the important details from the question.
    2. Reason through the evidence in the table and text step-by-step, looking for information to help you answer the question.
    3. Finally, output the final answer to the question in numerical or binary form without extra text.
Response Format:
- [Final Answer]: (YES, NO, or some number)
- [Justification]: The reasoning steps used that led to your final answer.
"""
            else:
                system_message = """You are an evaluation model tasked with analyzing the reasoning traces produced by a question-answering LLM.

You will be given:
1) A question
2) Evidence used for answering
3) A reasoning trace produced by an LLM
4) The LLM's predicted answer (YES / NO / some number), if present
5) The LLM's justification, if present
6) The ground-truth answer (YES / NO / some number)

Your task is to determine whether the reasoning trace contains any of the following failure types and, if so, whether they contributed to an incorrect final prediction.

IMPORTANT: Reasoning traces can contain no numerical reasoning failures. Do not infer failures unless there is clear evidence in the reasoning trace.

FAILURE TYPES
1) Numerical scoping error
- A numerical quantity is incorrectly bound to an entity, attribute, unit, population, or scope different from what is supported by the evidence.
- Examples: mixing up two different “rates,” attaching a number to the wrong country/year, confusing “revenue” vs “profit,” swapping numerator/denominator, applying a statistic to the wrong population, mismatching units (%, dollars, people, acres), or using evidence about one entity to justify a claim about another.

2) Arithmetic errors:
- The reasoning combines quantities incorrectly (wrong addition/subtraction/multiplication/division), miscomputes differences/percent changes, misapplies rounding, double-counts, or performs invalid aggregation.

3) Infinite loops / no final verdict:
The reasoning trace fails to terminate or does not produce a final answer due to repetitive reasoning.
- If a final answer is present and the trace ends cleanly, mark NO.

CAUSALITY RULE
For each failure type, you must decide whether it contributed to an incorrect final answer:
- If the predicted answer matches the ground-truth answer, any detected failure MUST be marked as not contributing to an incorrect answer.
- If the predicted answer does not match the ground-truth answer:
    - "yes" if the failure plausibly caused or materially contributed to the wrong answer.
    - "no" if the failure exists but is irrelevant to why the final answer is wrong.
If you are unsure about causality, choose "unclear" and explain why.

YOUR OUTPUT FORMAT (STRICT)
Return a single JSON object with exactly these top-level keys:
{
"verdict_correct": boolean,                 // whether predicted answer equals ground-truth answer (true or false)
"failures": {
    "numerical_scoping_error": {
    "exists": "yes"|"no",
    "where": [ { "span_quote": string, "explanation": string } ],
    "led_to_incorrect_answer": "yes"|"no"|"unclear"
    },
    "arithmetic_error": {
    "exists": "yes"|"no",
    "where": [ { "span_quote": string, "explanation": string } ],
    "led_to_incorrect_answer": "yes"|"no"|"unclear"
    },
    "infinite_loop": {
    "exists": "yes"|"no",
    "where": [ { "span_quote": string, "explanation": string } ],
    "led_to_incorrect_answer": "yes"|"no"|"unclear"
    }
},
"brief_summary": string                     // 1–3 sentences, focusing on the most important detected issue(s)
}

RULES
- If "exists" is "no", "where" MUST be an empty list [].
- "span_quote" should be a short direct quote (or two) from the reasoning trace or justification that localizes the failure.
- Keep explanations concrete: name the numbers, entities, units, and what was mismatched or miscomputed.
- Do not output anything except the JSON."""
        return system_message
    
    def get_user_message(self, query: str, evidence_list: List[str], fc_example_list: List[Dict[str, Any]], metadata: Dict[str, str]=None, judge: bool=False) -> str:
        """Gets the user message prompt."""
        if self.task == "fc":
            if not judge:
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
        else:
            if not judge:
                user_message = f"""Here are examples of question answering:
{self.get_input_examples(fc_example_list)}

Similar to the given examples, answer the following questions using the table and text evidence. Give the step by step solution justification. Output the final answer as binary (YES or NO) or as a number without additional text.
[Question]: \"{query}\"
[Evidences]: {self.get_evidences(evidence_list)}

Give the reply in the following format:
[Final Answer]: (YES, NO, or some number)
[Justification]:
"""
            else:
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
    
    def get_input_examples(self, fc_example_list: List[Dict[str, Any]], judge: bool=False) -> str:
        """Gets the string version of a list of fact checking examples."""
        input_examples = ""
        if self.task == "fc":
            if not judge:
                for idx, fc_example in enumerate(fc_example_list):
                    label = fc_example['label'] if fc_example['label'] not in self.label_map else self.label_map[fc_example['label']]
                    input_examples += f"{idx+1}. [Claim]: \"{fc_example['claim']}\"\n[Evidences]: {self.get_evidences(fc_example['evidence_list'])}\n[Label]: {label}\n\n"
            else:
                input_examples = """== INPUTS ==
[Claim]
"In the "do-nothing Senate," there are 352 House bills "sitting on Harry Reid's desk awaiting action," including 55 introduced by Democrats."

[Evidence]
"... Jenkins' comments, made at a press conference on July 29, 2014:

'352 bills are sitting on Harry Reid's desk, awaiting action.
98 percent of them passed with bipartisan support -- Republicans and Democrats working together to pass legislation.
50 percent of the bills passed unanimously, with no opposition.
70 percent of the bills passed with two-thirds support in the House.
And over 55 bills were introduced by Democrats.'

... We used the THOMAS congressional database to pluck out the 733 measures that have passed the House so far this year. 
What remained were 342 individual bills. That's 10 fewer than what Jenkins claimed, but we have no quarrel with her number, since the difference likely stems from additional bills being passed in the couple of days between when she spoke and when we looked through the database.

... It's an oversimplification to say that these bills are 'sitting on Harry Reid's desk.' Many have been assigned to committees.

... Do the House-passed, Senate-stalled bills include 55 introduced by Democrats? On the number, Jenkins is basically right -- we counted 54.

... The number of Democratic-sponsored bills Jenkins mentions -- 55 -- accounts for just 16 percent of all House-passed bills now awaiting consideration in the Senate.

... When we took a closer look at the 55 Democratic-sponsored bills that passed the House, we found that fewer than half -- 24 -- could reasonably be described as substantive.

... In some cases, committee chairs -- not Reid -- may be blocking or moving slowly on these bills.

... On balance, we rate the claim Half True."

[Fact-checking LLM Output: Reasoning Trace, LLM Predicted Label, and LLM Justification]
<think>
Alright, let me try to figure out how to assess this claim. The claim is that "In the 'do-nothing Senate,' there are 352 House bills 'sitting on Harry Reid's desk awaiting action,' including 55 introduced by Democrats." 

Looking at the evidence provided, there are several points. First, the evidence mentions that Jenkins claimed there are 352 bills on Reid's desk. However, when they checked the THOMAS database, they found 342 bills, which is 10 less than Jenkins' number. The difference is probably due to a couple of days between when Jenkins spoke and when the database was updated. So, Jenkins' number is slightly off, but not by much.

Next, the evidence discusses whether these bills are actually "sitting on Reid's desk." It says that many bills are assigned to committees, so they aren't necessarily waiting on the floor. Some might be fast-tracked or stalled by committee chairs. Also, the Senate might be working on similar bills without relying on the House bills as a starting point. 

There's a mention that Reid has influence over committee chairs, who can decide to fast-track or stall bills. Additionally, senators can threaten filibusters, which require 60 votes, making it hard to stop a bill. The evidence also points out that both parties have contributed to the gridlock, not just Reid or the Democrats.

Furthermore, when looking at the 55 Democratic-sponsored bills, only 16% are considered substantive, and most are minor or non-controversial. So, the claim that 55 introduced by Democrats are significant is overblown because many of these are not high-profile or meaningful legislation.

Putting it all together, the evidence shows that while there are a number of bills waiting in the Senate, the exact number Jenkins cited isn't accurate. The claim oversells the bipartisanship and the number of substantive bills, which are actually minor. Therefore, the claim isn't fully supported by the evidence.
</think>

[Label]: CONFLICTING

[Justification]: The evidence provides conflicting information regarding the number of bills on Reid's desk and their substantive nature. Jenkins claims 352 bills, but the database shows 342, differing by 10. Additionally, only 16% of the 55 Democratic bills are substantive, suggesting the claim overstates bipartisanship and bill significance.

[Ground Truth Label (Human)]
CONFLICTING
== END INPUTS ==

== YOUR OUTPUT ==
{
"verdict_correct": true,
"failures": {
    "numerical_scoping_error": {
    "exists": "yes",
    "where": [
        {
        "span_quote": "the exact number Jenkins cited isn't accurate",
        "explanation": "The numerical quantity was mis-scoped by treating an acknowledged approximate count as materially incorrect."
        },
        {
        "span_quote": "only 16% of the 55 Democratic bills are substantive",
        "explanation": "The reasoning incorrectly binds bill significance to a numerical claim about count."
        }
    ],
    "led_to_incorrect_answer": "no"
    },
    "arithmetic_error": {
    "exists": "no",
    "where": [],
    "led_to_incorrect_answer": "no"
    },
    "infinite_loop": {
    "exists": "no",
    "where": [],
    "led_to_incorrect_answer": "no"
    }
},
"brief_summary": "The model commits numerical scoping errors but still arrives at the correct final verdict."
}"""
        else:
            if not judge:
                for idx, fc_example in enumerate(fc_example_list):
                    answer = fc_example['answer'] if fc_example['answer'] not in self.label_map else self.label_map[fc_example['answer']]
                    input_examples += f"{idx+1}. [Question]: \"{fc_example['question']}\"\n[Evidences]: {self.get_evidences(fc_example['evidence_list'])}\n[Final Answer]: {answer}\n\n"
            else:
                input_examples = """== INPUTS ==
[Question]
"in 2010 and 2009 , what was the total fair value in billions of assets segregated for the benefit of securities and futures brokerage customers?"

[Evidence]
1. "Text Before Table:
- "jpmorgan chase & co./2010 annual report 273 the following table presents the u.s ."
- "and non-u.s ."
- "components of income before income tax expense/ ( benefit ) and extraordinary gain for the years ended december 31 , 2010 , 2009 and 2008 ."
- "year ended december 31 , ( in millions ) 2010 2009 2008 ."
"
2. "Table:
year ended december 31 ( in millions )|2010|2009|2008
u.s .|$ 16568|$ 6263|$ -2094 ( 2094 )
non-u.s. ( a )|8291|9804|4867
income before incometax expense/ ( benefit ) andextraordinary gain|$ 24859|$ 16067|$ 2773"
3. "Text After Table:
- "non-u.s. ( a ) 8291 9804 4867 income before income tax expense/ ( benefit ) and extraordinary gain $ 24859 $ 16067 $ 2773 ( a ) for purposes of this table , non-u.s ."
- "income is defined as income generated from operations located outside the u.s ."
- "note 28 2013 restrictions on cash and intercompany funds transfers the business of jpmorgan chase bank , national association ( 201cjpmorgan chase bank , n.a . 201d ) is subject to examination and regulation by the office of the comptroller of the currency ( 201cocc 201d ) ."
- "the bank is a member of the u.s ."
- "federal reserve sys- tem , and its deposits in the u.s ."
- "are insured by the fdic ."
- "the board of governors of the federal reserve system ( the 201cfed- eral reserve 201d ) requires depository institutions to maintain cash reserves with a federal reserve bank ."
- "the average amount of reserve balances deposited by the firm 2019s bank subsidiaries with various federal reserve banks was approximately $ 803 million and $ 821 million in 2010 and 2009 , respectively ."
- "restrictions imposed by u.s ."
- "federal law prohibit jpmorgan chase and certain of its affiliates from borrowing from banking subsidiar- ies unless the loans are secured in specified amounts ."
- "such secured loans to the firm or to other affiliates are generally limited to 10% ( 10 % ) of the banking subsidiary 2019s total capital , as determined by the risk- based capital guidelines ; the aggregate amount of all such loans is limited to 20% ( 20 % ) of the banking subsidiary 2019s total capital ."
- "the principal sources of jpmorgan chase 2019s income ( on a parent company 2013only basis ) are dividends and interest from jpmorgan chase bank , n.a. , and the other banking and nonbanking subsidi- aries of jpmorgan chase ."
- "in addition to dividend restrictions set forth in statutes and regulations , the federal reserve , the occ and the fdic have authority under the financial institutions supervisory act to prohibit or to limit the payment of dividends by the banking organizations they supervise , including jpmorgan chase and its subsidiaries that are banks or bank holding companies , if , in the banking regulator 2019s opinion , payment of a dividend would consti- tute an unsafe or unsound practice in light of the financial condi- tion of the banking organization ."
- "at january 1 , 2011 , jpmorgan chase 2019s banking subsidiaries could pay , in the aggregate , $ 2.0 billion in dividends to their respective bank holding companies without the prior approval of their relevant banking regulators ."
- "the capacity to pay dividends in 2011 will be supplemented by the banking subsidiaries 2019 earnings during the in compliance with rules and regulations established by u.s ."
- "and non-u.s ."
- "regulators , as of december 31 , 2010 and 2009 , cash in the amount of $ 25.0 billion and $ 24.0 billion , respectively , and securities with a fair value of $ 9.7 billion and $ 10.2 billion , respec- tively , were segregated in special bank accounts for the benefit of securities and futures brokerage customers ."
- "note 29 2013 capital the federal reserve establishes capital requirements , including well-capitalized standards for the consolidated financial holding company ."

[Fact-checking LLM Output: Reasoning Trace, LLM Predicted Label, and LLM Justification]
<think>
Alright, so I need to figure out the total fair value in billions of assets segregated for the benefit of securities and futures brokerage customers for 2010 and 2009. Let me go through the evidence step by step.

First, I'll look at the text before the table. It mentions that the information is from the 2010 annual report of JPMorgan Chase & Co. It talks about segregated assets for the benefit of securities and futures brokerage customers. 

Next, I check the table provided. The table has three columns: 2010, 2009, and 2008, with values in millions. The columns are labeled as "u.s." and "non-u.s." and then "income before income tax expense/ ( benefit ) and extraordinary gain."

Looking at the 2010 row, the "u.s." value is $16,568 million, and the "non-u.s." is $8,291 million. The total income before tax and gain is $24,859 million. 

But wait, the question is about the total fair value of segregated assets, not the income. The table mentions "segregated in special bank accounts for the benefit of securities and futures brokerage customers" in the text after the table. Specifically, it states that as of December 31, 2010 and 2009, the fair value of these segregated assets was $9.7 billion and $10.2 billion, respectively.

So, for 2010, it's $9.7 billion, and for 2009, it's $10.2 billion. The question asks for the total in billions, so I just need to add these two amounts together.

Adding $9.7 billion and $10.2 billion gives me $19.9 billion. Since the question asks for the total in billions, the answer is 19.9, but since we're to provide it without extra text, I'll write it as a number.

Wait, but looking back, the table's numbers are in millions. The table shows $16,568 million for U.S. and $8,291 million for non-U.S. But the text after the table mentions the fair value of segregated assets as $9.7 billion and $10.2 billion. So the table's numbers are about income, not the segregated assets themselves. Therefore, the segregated assets' fair value is given directly in the text, not in the table.

So the total fair value for 2010 is $9.7 billion, and for 2009, it's $10.2 billion. Adding them together gives $19.9 billion, which is 19.9 in billions. But the question asks for the total in billions, so the answer is 19.9, but since the user might expect a whole number, perhaps it's rounded to 20. However, the exact figures are 9.7 and 10.2, so the precise total is 19.9.

But looking again, the question is about the total fair value in billions for both years. So I think the answer is 19.9, but since the user might prefer it as a whole number, maybe 20. However, the exact calculation is 9.7 + 10.2 = 19.9, so I'll go with 19.9.

Wait, but the user's examples sometimes round to one decimal place. So 19.9 is correct. Alternatively, if the answer expects a whole number, it might be 20, but I think 19.9 is accurate based on the given data.
</think>

[FInal Answer]: 19.9

[Justification]: The total fair value of assets segregated for the benefit of securities and futures brokerage customers in 2010 was $9.7 billion, and in 2009, it was $10.2 billion. Adding these together gives $19.9 billion.

[Ground Truth Answer]
68.9
== END INPUTS ==

== YOUR OUTPUT ==
{
"verdict_correct": false,
"failures": {
    "numerical_scoping_error": {
    "exists": "yes",
    "where": [
        {
        "span_quote": "total fair value for 2010 is $9.7 billion",
        "explanation": "Mis-scoped the numerical quantity for 2010 as only 9.7 billion dollars, missing the $25.0 billion."
        },
        {
        "span_quote": "2009, it's $10.2 billion",
        "explanation": "Mis-scoped the numerical quantity for 2009 as only 10.2 billion dollars, missing the $24.0 billion."
        }
    ],
    "led_to_incorrect_answer": "yes"
    },
    "arithmetic_error": {
    "exists": "no",
    "where": [],
    "led_to_incorrect_answer": "no"
    },
    "infinite_loop": {
    "exists": "no",
    "where": [],
    "led_to_incorrect_answer": "no"
    }
},
"brief_summary": "The model commits numerical scoping errors which cause an incorrect final verdict."
}"""
        
        return input_examples