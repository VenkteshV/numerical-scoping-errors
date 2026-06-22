# Investigating Scoping Errors in Open-Domain Numerical Reasoning

<img src="./assets/banner.png" alt="Scoping landscape banner" style="width:100%; height:auto;">

A framework for generating LLM responses for numerical fact-checking and question answering tasks, detecting scoping errors at a statistic- and trace-level, distilling that scoping signal into smaller student verifiers, and using scoping signals to improve downstream performance through test-time scaling.

## Overview

This repository provides code for:

1. Prompting pre-trained LLMs to either fact-check complex numerical claims or answer multi-step numerical questions using evidence.

<img src="./assets/pipeline.png" alt="Inference pipeline" style="width:70%; height:auto;">

2. Detecting scoping errors in fact-checking or QA reasoning traces at a statistic- and trace-level using a hybrid LLM-as-a-Judge pipeline.

<img src="./assets/judge_pipeline.png" alt="Hybrid LLM-as-a-Judge pipeline" style="width:70%; height:auto;">

3. Distilling correctness and scoping error signals into outcome reward models.

<img src="./assets/distillation.png" alt="Distillation into Scoping-Aware ORMs" style="width:100%; height:auto;">

4. Using detected scoping error signals to select candidate LLM reasoning traces in test-time scaling.

<img src="./assets/variants.png" alt="Scoping-aware mitigation variants" style="width:80%; height:auto;">

#### Key Features:

- Multi-model support (all inference LLMs are hosted through LMStudio, thus any LLM hosted in LMStudio can be used for inference or judging)
- Test-time scaling support (all inference code can be scaled to process queries multiple times to generate candidate responses and judgements)
- Multi-task support (implementation supports both numerical fact-checking and numerical question-answering)
- Modular architecture (LLMs and CoreNLP quantity extraction services are hosted externally allowing for modular and customizable set-up)

#### Research Context:

This implementation is part of Master's thesis research on improving the reasoning capabilities of Large Language Models for real-world open-domain numerical tasks. The framework specifically investigates numerical scoping errors in complex tasks that require multi-step reasoning and numerical understanding. Here is the abstract of the associated thesis:

> Numerical information shapes how people interpret real-world events, evaluate claims, and make decisions. As Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly used for open-domain information seeking, their ability to reason reliably over quantities has become critical. Yet, even when models retrieve relevant evidence and produce fluent chain-of-thought reasoning, they still often reach incorrect conclusions by failing to reason about numbers in the right contexts. This work investigates _numerical scoping errors_, an underexplored reasoning failure where a model incorrectly binds a quantity to the wrong referent, time-frame, unit, population, condition, or source. In high impact settings, a single mis-scoped number can change the meaning of a conclusion, turning a statistic into a misleading financial signal, an unsafe medical recommendation, or a false public claim. This makes numerical scoping errors an important reasoning failure to investigate. When performing numerical fact-checking (FC) and question-answering (QA), this work identifies scoping errors as a common and consequential failure mode in generated LLM reasoning traces across tasks. In human analysis, scoping errors appeared in 44.7\% of numerical FC and 26.8\% of numerical QA traces, with 35.3\% and 81.8\% of these errors, respectively, judged to contribute to incorrect final answers. To detect these failures at scale, this work develops a hybrid LLM-as-a-Judge pipeline that decomposes reasoning traces into quantities, verifies whether each quantity is correctly scoped based on its source type, and aggregates these quantity-level judgements into trace-level labels. The pipeline achieves moderate agreement with human annotations, but can be expensive to leverage. To explore more efficient alternatives, the hybrid pipeline's scoping detection signal is distilled into smaller, conservative student models which process traces around 212 times faster. Finally, this work evaluates whether scoping-aware reward signals can improve downstream performance in a parallel test-time scaling setting, where the system chooses among multiple generated reasoning traces, producing modest improvements, especially when combined with correctness-based selection. Overall, this work shows that numerical reasoning failures are not merely errors of calculation but errors of context. By making these contextual failures visible and measurable, this work takes a step toward numerical reasoning systems that can support high-impact, real-world decisions not only because they calculate correctly, but because they understand what their numbers mean.

## Repository Structure

```
root/
├── code/
│   ├── main.py                                      # Code entrypoint and argument handler
│   ├── numerical_reasoning_handler.py               # Main engine for inference
│   ├── scoping_error_measurer.py                    # Main engine for scoping error detection
│   ├── llm/
│   │   ├── llm_handler.py                           # Handler for LMStudio inference
│   |   ├── system_prompts.json                      # System prompts used for different tasks
│   |   └── Inference configurations via completion_params_*.json
│   ├── utils/
│   |   └── corenlp_quantity_segmenter.py            # Connection to external CoreNLP service
│   ├── validation/
│   |   └── validation_handler.py                    # Handler for inference validation
│   └── verifier/
│       ├── training_code/
│       |   └── Code and data for training student verifiers
│       ├── multitask_verifier_handler.py            # Handler for multi-task verifier inference
│       ├── verifier_handler.py                      # Handler for single-task verifier inference
│       └── Inference configurations via config_*.toml
├── data/
│   ├── finqa/
│   │   └── Financial Question-Answering questions, answers, evidences, and extracted quantities; train & test splits
│   ├── quantemp/
│       └── Numerical Fact-Checking claims, labels, evidences, and extracted quantities; train & test splits
└── output/
    └── Generated inference results and evaluations
```

## Set-up Instructions

### Prerequisites

- CUDA-compatible GPU with at least 16 GB of RAM (training ORMs will require even more)
- System capable of running Python and Java

### 1. Clone this repository
```
git clone https://github.com/msmink01/numerical-scoping-errors.git
cd numerical-scoping-errors/code
```

### 2. Create the main virtual environment and install dependencies
```
conda create -n scoping python=3.10.9
conda activate scoping
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 3. Set up LMStudio

  1. Install [LMStudio](https://lmstudio.ai/) and download the `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `lmstudio-community/Qwen2.5-14B-Instruct-GGUF` and `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` models

      _Note: Other LLMs can be switched in as desired._

  2. If performing reasoning trace generation, load the `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model in LMStudio and change the `Load` `Context Length` to `8192`.

  3. If performing LLM-as-a-Judge judgements, load the `lmstudio-community/Qwen2.5-14B-Instruct-GGUF` model in LMStudio and change the `Load` `Context Length` to `16384`.

  4. Ensure the Local Inference Server is enabled with `Status` `Running` in the `Developer` tab (should be running on `localhost:1234`)

### 4. Set up the CoreNLP service (if using the Hybrid LLM-as-a-Judge pipeline to detect scoping errors)

  1. Install a Java Development Kit if not already installed (I installed version 25.0.2 from Oracle [here](https://www.oracle.com/java/technologies/downloads/#jdk25-windows))

      Verify installation by running `java -version` in a separate CLI window.
      
      
      If `java` is not found, set environment variable: `JAVA_HOME` to your JDK folder and add `JAVA_HOME` to your `PATH`.
    
  
  2. Download and unzip the most recent [Stanford CoreNLP jar](https://stanfordnlp.github.io/CoreNLP/download.html)
  
      After extraction, files such as `stanford-corenlp-*.jar` and `stanford-corenlp-*-models.jar` should be in the folder.
  
  3. Run CoreNLP as a service from extracted folder (in a new terminal window not running the `scoping-thesis` virtual environment above)
  
  ```
  java -mx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 600000
  ```
  Should see a line that says: `[main] INFO CoreNLP - StanfordCoreNLPServer listening at /[0:0:0:0:0:0:0:0]:9000`

  The CoreNLP server is now started up.


## Usage

The following commands are all run from the `code` directory.

### Standard Inference

Run basic fact-checking or question-answering inference with a desired dataset and evidence split:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test_[finqa|quantemp].json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr]
```
_Note: make sure the correct LLM is loaded into LMStudio for inference!_

Parameters:
- `input-json-file`: path to json file that contains input queries, few-shot examples, evidences, and answers
- `output-json-file`: path to json file that will contain all inference outputs
- `task`: `fc` for fact-checking or `qa` for question-answering
- `split`: which evidence split to use for each dataset, must match the evidence split in the input json file

### Straight LLM-as-a-Judge Failure Detection
Run straight LLM-as-a-Judge processing to find scoping, arithmetic, and looping errors in reasoning traces:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test_[finqa|quantemp].json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --judge True
```
_Note: output json must match existing json file which already contains standard inference outputs. Ensure the judge LLM is loaded in LMStudio!_

Parameters:
- `input-json-file`: path to json file that contains input queries, few-shot examples, evidences, and answers
- `output-json-file`: path to json file that will contain all inference outputs
- `task`: `fc` for fact-checking or `qa` for question-answering
- `split`: which evidence split to use for each dataset, must match the evidence split in the input json file
- `judge`: whether to perform straight LLM-as-a-Judge inference

### Hybrid LLM-as-a-Judge Pipeline Scoping Failure Detection
Run hybrid LLM-as-a-Judge pipeline to find scoping errors at a statistic- and trace-level in reasoning traces:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test_[finqa|quantemp].json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --advanced-judge True --repair True
```
_Note: output json must match existing json file which already contains standard inference outputs. Ensure the judge LLM is loaded in LMStudio! Ensure the CoreNLP service is running!_

Parameters:
- `input-json-file`: path to json file that contains input queries, few-shot examples, evidences, and answers
- `output-json-file`: path to json file that will contain all inference outputs
- `task`: `fc` for fact-checking or `qa` for question-answering
- `split`: which evidence split to use for each dataset, must match the evidence split in the input json file
- `advanced-judge`: whether to perform hybrid LLM-as-a-Judge pipeline to detect scoping errors
- `repair`: whether to try to repair any broken LLM-as-a-Judge calls in processing with a larger context length

### Single-Task Outcome Reward Model Correctness/Scoping Scoring
Run distilled single-task ORMs to judge reasoning traces for general correctness and scoping quality:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test_[finqa|quantemp].json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --verifier verifier/[desired config name].toml
```
_Note: output json must match existing json file which already contains standard inference outputs. Ensure the verifier path is correct and the model weights exist!_

Parameters:
- `input-json-file`: path to json file that contains input queries, few-shot examples, evidences, and answers
- `output-json-file`: path to json file that will contain all inference outputs
- `task`: `fc` for fact-checking or `qa` for question-answering
- `split`: which evidence split to use for each dataset, must match the evidence split in the input json file
- `verifier`: whether to use single-task ORMs to score reasoning traces for correctness or scoping quality; must be the path to the `.toml` file containing the single-task variants

### Multi-Task Outcome Reward Model Correctness/Scoping Scoring
Run distilled multi-task ORMs to judge reasoning traces for general correctness and scoping quality:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test_[finqa|quantemp].json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --verifier-multitask verifier/[desired config name].toml
```
_Note: output json must match existing json file which already contains standard inference outputs. Ensure the verifier path is correct and the model weights exist!_

Parameters:
- `input-json-file`: path to json file that contains input queries, few-shot examples, evidences, and answers
- `output-json-file`: path to json file that will contain all inference outputs
- `task`: `fc` for fact-checking or `qa` for question-answering
- `split`: which evidence split to use for each dataset, must match the evidence split in the input json file
- `verifier-multitask`: whether to use multi-task ORMs to score reasoning traces for correctness and scoping quality; must be the path to the `.toml` file containing the multi-task variants

### Test-Time Scaling
To perform any of the above tasks with test-time scaling, meaning each query will be processed multiple times, make the following changes to the CLI command:

1. Append the `--completion-params` parameter which describes how many times each LLM temperature should generate a new candidate trace.  

2. Change the `--output-json-file` to point to a directory instead of a specific json file

For example:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test.json --output-json-file outputs/tts_outputs/ --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --completion-params 3*llm/tts_completion_params/temp_0.2.json 3*llm/tts_completion_params/temp_0.3.json 3*llm/tts_completion_params/temp_0.4.json 3*llm/tts_completion_params/temp_0.5.json
```
will perform standard inference for 12 reasoning traces per one query: 3 traces with an LLM temperature of `0.2`, 3 with `0.3`, etc. These outputs will be stored in `outputs/tts_outputs/0.2_0.json`, `.../0.2_1.json`, `.../0.2_2.json`, `.../0.3_0.json`, etc.

_Note: temperatures can be customized by changing the `llm/tts_completion_params/*.json` files._

The evaluation of all test-time scaling variants can be run through the `validation/tts_validation_handler.py` file.

### Custom Datasets

To use your own data, create a new `../data/*.json` file with matching formats for each specific task. Furthermore, any evidence split can be added to any existing files by creating a new dictionary key containing the associated evidence and changing the `--split` parameter.

## Output Format

Outputs are returned in single json files defined by the `--output-json-file` parameter. Output json files can contain the following keys:

```
{
    "input_json_file": "...",                          # Input json file used to generate these results, defined by --input-json-file parameter
    "evidence_split": "...",                           # Evidence split used to generate these results, defined by --split parameter
    "outputs": [ {                                     # Contains the general inference results of accomplishing each query in the task
        "repsonse": " ... ",                           # Exact string response returned by general inference model for the i-th query in the input json file
        "total_time": float,                           # Time taken to get response
    }, ... ],
    "validation": { ... },                             # General inference validation results, change depending on the task type: EM accuracies, F1 scores, and invalidity rates
    "judge": [ { ... }, ...],                          # Straight LLM-as-a-Judge inference results for detecting scoping, arithmetic, and looping errors
    "judge_validation": { ... },                       # Straight LLM-as-a-Judge validation results, mainly counts of error prevalence and pestilence
    "advanced_judge": [ {                              # Contains the Hybrid LLM-as-a-Judge pipeline results for detecting scoping errors at statistic- and trace-levels for each trace
        "valid": True|False,                           # Whether this judge response returned a valid result
        "raw_scoping_error": True|False,               # Whether this reasoning trace contained a scoping error in any statistic, even the non-important ones
        "filtered_scoping_error": True|False,          # Whether this reasoning trace contained a scoping error in an important statistic (DEFAULT)
        "per_class": {                                 # Counts of each type of statistic and how many were mis-scoped
            "extractive": {'n': int, 'n_correct': int, 'n_incorrect': int, 'n_unknown': int},
            "extrapolative": { ... },
            "parametric": { ... },
            "unknown": { ... },
        },
        "errors": [ { ... }, ... ],                    # List of all the statistics that were mis-scoped
        "unknowns": [ { ... }, ... ],                  # List of all the statistics that had an unknown type or unknown scoping label
        "thinking_quantities": {                       # All extracted thinking statistics, their de-duplication attributes, type, and scoping correctness labels
            "4187.8": [ {
                "quantity_texts": [ "$ 4187.8 million", ... ],    # List of mentions of the quantity in thinking
                "sentence_idxs": [ int, ... ],                    # List of sentence idxs where the quantity appears
                "sentence_texts": [ " ... ", ... ],               # List of sentences where quantity is mentioned
                "quantity_metadata": {                            # Attributes used for quantity de-duplication and matching with input statistics
                    "type": MONEY,
                    "value": 4187800000.0,
                    "unit": "MONEY",
                    "approx": False,
                    "tail": "$",
                    "text": "$ 4187.8 million",
                },
                "type": "extractive",                             # The type of statistic as identified by pipeline: extractive, extrapolative, parametric, or unknown
                "correct": "true",                                # Whether the statistic is scope-valid (true), mis-scoped (false), or unknown
                "check": { ... },                                 # Exact json provided by the LLM-judge call to decide the scoping label
                "importance": float,                              # Importance weight given by the importance LLM-judge
                "importance_check": { ... },                      # Exact json provided by the LLM-judge call to decide the quantity importance
            }, ... other versions of 4187.8 that is not monetary ],
            ...
        },
        "response_dicts":  {                             # Exact response dicts for each step in the hybrid LLM-as-a-Judge pipeline
            "extractive": {
                "str_response": [ " ... ", ... ],        # List of all str responses for this type of judge call for each block of 20 statistics
                "time": [ float, ... ],                  # List of all times for each judge call for each block of 20 statistics
                "results": [ true|false|unknown, ... ],  # List of all scoping labels for each statistic across all judge calls
                "json": { ... },                         # Full result json parsed across all judge calls
                "valid_json": [ True|False, ... ],       # Whether each returned json was parseable
                "valid": True|False,                     # Whether entire step of the hybrid LLM-as-a-Judge was valid
            },
            "non_extractive": { ... },
            "importance": { ... },
        }, 
    }, ... ],
    "verifier": {                                        # Verifier scores for each reasoning trace depending on config toml files
        "verifier/config_[finqa|quantemp].toml": [ {     # Single-task verifier scores for trace correctness/scoping quality
            "score": float,                              # Verifier score for a trace without the thinking included in inputs
            "score_thinking": float,                     # Verifier score for a trace WITH thinking included in inputs (DEFAULT)
            "total_time": float,                         # Time taken for verifier to return a response
        }, ... ],
        "verifier/config_[finqa|quantemp]_multitask.toml": [ {
            "score": float,                              # Correctness verifier score for a trace without the thinking included in inputs
            "score_thinking": float,                     # Correctness verifier score for a trace WITH thinking included in inputs (DEFAULT)
            "scoping_score": float,                      # Scoping verifier score for a trace without the thinking included in inputs
            "scoping_score_thinking": float,             # Scoping verifier score for a trace WITH thinking included in inputs (DEFAULT)
            "total_time": float,                         # Time taken for verifier to return a response
        }, ... ],
    },
}
```

## References

1. [Thesis Report](https://repository.tudelft.nl/record/uuid:919c8be0-b6a1-41df-bf05-ae46f4c51f86)
2. [QUANTEMP Dataset](https://arxiv.org/abs/2403.17169)
3. [FinQA Dataset](https://arxiv.org/abs/2109.00122)
4. [LMStudio](https://lmstudio.ai/)
5. [Deepseek models](https://arxiv.org/abs/2412.19437)

## Citation

This project is part of academic research at TU Delft. Please contact the author for questions, collaboration, or usage permissions. If you use this thesis in your work, please cite:

```
@mastersthesis{Smink2026,
  author       = {Moniek Smink},
  title        = {Investigating Scoping Errors in Open-Domain Numerical Reasoning},
  school       = {Delft University of Technology},
  year         = {2026},
  month        = {June},
  note         = {Master's thesis, Electrical Engineering, Mathematics and Computer Science},
  url          = {https://repository.tudelft.nl/record/uuid:919c8be0-b6a1-41df-bf05-ae46f4c51f86}
}
```
