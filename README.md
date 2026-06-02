# Investigating Scoping Errors in Open-Domain Numerical Reasoning

A framework for generating LLM responses for numerical fact-checking and question answering tasks, detecting scoping errors at a statistic- and trace-level, distilling that scoping signal into smaller student verifiers, and using scoping signals to improve downstream performance through test-time scaling.

## Overview

This repository provides code for:

1. Prompting pre-trained LLMs to either fact-check complex numerical claims or answer multi-step numerical questions using evidence.
2. Detecting scoping errors in fact-checking or QA reasoning traces at a statistic- and trace-level using a hybrid LLM-as-a-Judge pipeline.
3. Distilling correctness and scoping error signals into outcome reward models.
4. Using detected scoping error signals to select candidate LLM reasoning traces in test-time scaling.


#### Key Features:

- Multi-model support (all inference LLMs are hosted through LMStudio, thus any LLM hosted in LMStudio can be used for inference or judging)
- Test-time scaling support (all inference code can be scaled to process queries multiple times to generate candidate responses and judgements)
- Multi-task support (implementation supports both numerical fact-checking and numerical question-answering)
- Modular architecture (LLMs and CoreNLP quantity extraction services are hosted externally allowing for modular and customizable set-up)

#### Research Context:

This implementation is part of Master's thesis research on improving the reasoning capabilities of Large Language Models for real-world open-domain numerical tasks. The framework specifically investigates numerical scoping errors in complex tasks that require multi-step reasoning and numerical understanding.

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
│   ├── quantemp++/
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
git clone https://github.com/msmink01/overanalysis-thesis.git
cd overanalysis-thesis/code
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

### Standard Inference

Run basic fact-checking or question-answering inference with a desired dataset and evidence split:
```
python main.py --input-json-file ../data/[finqa|quantemp]/test.json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr]
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
python main.py --input-json-file ../data/[finqa|quantemp]/test.json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --judge True
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
python main.py --input-json-file ../data/[finqa|quantemp]/test.json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --advanced-judge True --repair True
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
python main.py --input-json-file ../data/[finqa|quantemp]/test.json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --verifier verifier/[desired config name].toml
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
python main.py --input-json-file ../data/[finqa|quantemp]/test.json --output-json-file outputs/output.json --task [fc|qa] --split [gold|reranked|qrel|whole_document|fsr] --verifier-multitask verifier/[desired config name].toml
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

### Custom Datasets

To use your own data, create a new `../data/*.json` file with matching formats for each specific task.

## Output Format

## References

## License

This project is part of academic research at TU Delft. Please contact the author for licensing information and usage permissions.
