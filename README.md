# Scoping Error Thesis

## Set-up Instructions

### 1. Clone this repository

### 2. Create the main virtual environment and install dependencies

### 3. Set up the CoreNLP service

  1. Install a Java Development Kit if not already installed (I installed version 25.0.2 from Oracle [here](https://www.oracle.com/java/technologies/downloads/#jdk25-windows))

  
  Verify installation by running `java -version`.
  
  
  If `java` is not found, set environment variable: `JAVA_HOME` to your JDK folder and add `JAVA_HOME` to your `PATH`.
    
  
  2. Download and unzip the most recent [Stanford CoreNLP jar](https://stanfordnlp.github.io/CoreNLP/download.html)
  
  After extraction, files such as `stanford-corenlp-*.jar` and `stanford-corenlp-*-models.jar` should be in the folder.
  
  3. Run CoreNLP as a service from extracted folder (in a new terminal window not running the `scoping-thesis` virtual environment above)
  
  ```
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
  ```

  Should see a line that says: `[main] INFO CoreNLP - StanfordCoreNLPServer listening at /[0:0:0:0:0:0:0:0]:9000`

  The CoreNLP server is now started up.

  ### 4. Set up the QA_NatVer service

  1. Set up the [`QA_NatVer` repository](https://github.com/Raldir/QA-NatVer/tree/public_repo) as a submodule

  Run: `git submodule update --init --recursive`

  2. Download the required [model](https://drive.google.com/file/d/1fSDA7rlp39tEDAehoN8EeybEf9wSqEBo/view?usp=sharing) and [aligner](https://drive.google.com/file/d/1-391W3NKTlJLMpjUwjCv03H1Qk0ygWzC/view?usp=sharing) as instructed in the README.

  Place the model in the `third_party/QA-NatVer/models/BART0` folder. Should contain a file called: `finish.pt`
  
  Place the aligner in the `third_party/QA-NatVer/models/awesomealign` folder. Should be a directory called: `default_gold_no_nei_few_shot_False_4000`

  3. In a new CLI, create an entirely new virtual environment according to `QA-NatVer`'s README

  ```
  conda create -n qa_natver python=3.9
  conda activate qa_natver
  ```

  Then install dependencies with: `python -m pip install -e .` and `python -m pip install -U fastapi uvicorn pydantic==1.10.26`

  Verify working installation with: 
  ```
  python -c "import numpy as np; import matplotlib; print(np.__version__, matplotlib.__version__)"
  python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda version', torch.version.cuda); print('devices', torch.cuda.device_count())"
  ```

  Note: if you run into issues with numpy version conflicts between 1.x and 2.x with matplotlib
  ```
  pip install --upgrade --force-reinstall "numpy==1.26.4"
  pip install "matplotlib==3.6.3" --no-deps
  ```

  Note: if you run into issues with a GPU not being available with torch, re-install a CUDA version of torch [from here](https://pytorch.org/get-started/previous-versions/)
  ```
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  ```

  4. Verify that the repository works with a toy example

  Inside the `third_party/QA-NatVer` repository, run the following command which takes `data/test.jsonl` as input: 
  ```python -m src.run -c configs/alignment/dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev.json+configs/dataset/fever.json+configs/environment/local_saving.json+configs/model/bart0_trained.json+configs/samples/32.json -k exp_name=fever/environment_local_saving_model_bart0_trained_alignment_dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev_samples_32/0 seed=0
  ```

  Note: if you run into issues relating to pickling and `collate_fn`, try setting `"num_workers": 8` to `"num_workers": 0` in `QA-NatVer/src/utils/Config.py`

  5. Finally, host the QA-NatVer service

  Inside the `third_party/QA-NatVer` repository, set the `PYTHONPATH` to allow uvicorn to find the `qa_natver_server.py` file: `set PYTHONPATH=..;%PYTHONPATH%`

  Then start the uvicorn server: `python -m uvicorn qa_natver_server:app --host 127.0.0.1 --port 8008 --reload`

  Verify the server with a simple example (first for Powershell, second for bat): 
  ```{shell}
  $body = @{
    items = @(
      @{
        id = 1
        claim = "The evidence mentions the judiciary's allocation as Ksh554 million"
        evidence = @("with regard to the judicial governance of the judiciary, the superior billion, with matters settled having a value of ksh554 million.")
      }
    )
  } | ConvertTo-Json -Depth 5
  
  Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8008/infer" `
    -ContentType "application/json" `
    -Body $body
  ```
  ```{bat}
  curl -X POST "http://127.0.0.1:8008/infer" ^
    -H "Content-Type: application/json" ^
    -d "{\"items\":[{\"id\":1,\"claim\":\"The evidence mentions the judiciary's allocation as Ksh554 million\",\"evidence\":[\"with regard to the judicial governance of the judiciary, the superior billion, with matters settled having a value of ksh554 million.\"]}]}"
  ```




  
