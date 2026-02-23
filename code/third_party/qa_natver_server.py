import os
import sys
import gc
import json
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

# From QA-NatVer repo: set PYTHONPATH=..;%PYTHONPATH%
# python -m uvicorn qa_natver_server:app --host 127.0.0.1 --port 8008 --reload
# Then in other window: 
# $body = @{
#   items = @(
#     @{
#       id = 1
#       claim = "The evidence mentions the judiciary's allocation as Ksh554 million"
#       evidence = @("with regard to the judicial governance of the judiciary, the superior billion, with matters settled having a value of ksh554 million.")
#     }
#   )
# } | ConvertTo-Json -Depth 5

# Invoke-RestMethod -Method Post `
#   -Uri "http://127.0.0.1:8008/infer" `
#   -ContentType "application/json" `
#   -Body $body

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

# Resolve path to third_party/QA-NatVer 
CURRENT_DIR = Path(__file__).resolve().parent
QANATVER_ROOT = CURRENT_DIR / "QA-NatVer"

# Add QA-NatVer root to python path if not already there
if str(QANATVER_ROOT) not in sys.path:
    sys.path.insert(0, str(QANATVER_ROOT))

# QA-NatVer src/run.py imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.tfew import EncoderDecoder
from src.utils.Config import Config
from src.data.data_module_joint import FinetuneDataModuleJoint
from src.constants import CHOICES_SCORES, IDX

app = FastAPI(title="QA-NatVer Inference Server", version="0.1")

class Item(BaseModel):
    id: int
    claim: str
    evidence: List[str]

class InferRequest(BaseModel):
    items: List[Item]

class InferResponse(BaseModel):
    results: List[Dict[str, Any]]

# ---- Global model state ----
MODEL: Optional[EncoderDecoder] = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_lock = threading.Lock()

# ---- Global data state ----
REPO_DATA_DIR = QANATVER_ROOT / "data"
REPO_TEST_JSONL = REPO_DATA_DIR / "test.jsonl"
REPO_TEST_PROCESSED = REPO_DATA_DIR / "test_processed.jsonl"

_request_lock = threading.Lock()  # IMPORTANT: prevent concurrent requests

# cache your config path / overrides once
CONFIG_FILES = (
    "alignment/dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev.json+"
    "dataset/fever.json+"
    "environment/local_saving.json+"
    "model/bart0_trained.json+"
    "samples/32.json"
)
CONFIG_OVERRIDES = {"seed": "0"}

def load_qanatver_model(config_files: str, extra_kwargs: Optional[dict] = None):
    """
    Load QA-NatVer model weights once.
    Use the same config mechanism QA-NatVer already uses, but DO NOT run Trainer/test/evaluator.
    """
    global MODEL, TOKENIZER

    extra_kwargs = extra_kwargs or {}

    config = Config(config_files, extra_kwargs)  # QA-NatVer config loader

    # mimic get_transformer() from src/run.py (tokenizer + base seq2seq model)
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)
    tokenizer.model_max_length = config.max_seq_len

    model = EncoderDecoder(config, tokenizer, base_model)

    assert config.load_weight, "config.load_weight must point to your trained weights (e.g. models/BART0/finish.pt)"
    state = torch.load(config.load_weight, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()

    TOKENIZER = tokenizer
    MODEL = model

def unload_model():
    global MODEL, TOKENIZER

    with _model_lock:
        MODEL = None
        TOKENIZER = None

        # Free Python + GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def ensure_model_loaded(force_reload: bool = False):
    global MODEL, TOKENIZER

    if MODEL is not None and TOKENIZER is not None and not force_reload:
        return  # already loaded

    with _model_lock:
        # another thread might have loaded while we waited
        if MODEL is not None and TOKENIZER is not None and not force_reload:
            return

        # if forcing reload, unload first
        if force_reload:
            unload_model()

        load_qanatver_model(CONFIG_FILES, extra_kwargs=CONFIG_OVERRIDES)

def get_test_dataloader(mode: str):
    """
    mode: "natop" or "verdict"
    """
    config = Config(CONFIG_FILES, CONFIG_OVERRIDES)
    dm = FinetuneDataModuleJoint(config, TOKENIZER, mode=mode)

    # This triggers DatasetProcessor(split="test") internally, which reads data/test.jsonl
    dm.setup(stage="test")
    dl = dm.test_dataloader()

    return config, dl

def write_repo_test_jsonl(items: List[Item]) -> None:
    REPO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Always overwrite (because DatasetProcessor(split="test") reads this exact file)
    with REPO_TEST_JSONL.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "id": it.id,
                "claim": it.claim,
                "evidence": it.evidence,
                # optional; if omitted it defaults to SUPPORTS in DatasetProcessor
                # "label": "SUPPORTS",
            }) + "\n")

    # Optional: remove stale processed file so you don't accidentally reuse old processed output
    if REPO_TEST_PROCESSED.exists():
        try:
            REPO_TEST_PROCESSED.unlink()
        except OSError:
            pass

def infer_mode(mode: str) -> List[Dict[str, Any]]:
    """
    Returns a dict keyed by item id: {id: {"probs": [...], "pred": "...", "pred_idx": int}}
    """
    config, dl = get_test_dataloader(mode)

    id_to_out: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dl:
            # move batch tensors to device
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            out = MODEL(batch)

            # out[IDX] is a list of ids (same dedupe key used by LitModule epoch_end)
            ids = out["idx"] # query id
            preds = out["prediction"] # index of the best prediction
            probs_list = out["prediction_prob_list"] # probability of the best prediction
            scores_list = out["choices_scores_list"] # probabilities of all the possible predictions

            for i, qid in enumerate(ids):
                qid_int = int(qid)
                pred_idx = int(preds[i]) if not isinstance(preds[i], (list, dict)) else preds[i]
                label_map = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"] if mode == "verdict" else ["NO", "YES"]
                pred_label = label_map[pred_idx] if pred_idx < len(label_map) else str(pred_idx)

                id_to_out.append({
                    "probs": float(probs_list[i]),
                    "scores": [float(x) for x in scores_list[i]],
                    "pred_idx": int(pred_idx),
                    "pred_label": str(pred_label),
                })

    return id_to_out

# # On start-up model loader
# @app.on_event("startup")
# def startup():
#     # Example: same config composition you used on CLI:
#     # configs/alignment/... + configs/dataset/... + configs/environment/... + configs/model/... + configs/samples/...
#     config_files = (
#         "configs/alignment/dynamic_awesomealign_bert_mwmf_coarse_finetuned_4000_gold_no_nei_only_retrieval_5_ev.json+"
#         "configs/dataset/fever.json+"
#         "configs/environment/local_saving.json+"
#         "configs/model/bart0_trained.json+"
#         "configs/samples/32.json"
#     )
#     # Make sure load_weight path exists for serving.
#     load_qanatver_model(config_files, extra_kwargs={"seed": 0})

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    ensure_model_loaded(force_reload=False)
    assert MODEL is not None, "Model not loaded"
    with _request_lock:
        # 1) write request items to QA-NatVer/data/test.jsonl
        write_repo_test_jsonl(req.items)

        # 2) run natop + verdict forward inference (no evaluator)
        natop = infer_mode("natop")
        verdict = infer_mode("verdict")

    # merge outputs per id (keep original request ordering)
    results = []
    for i, it in enumerate(req.items):
        results.append({
            "id": it.id,
            "verdict": verdict[i],
        })

    return {"results": results}

@app.post("/unload")
def unload():
    unload_model()
    return {"ok": True, "message": "Model unloaded and GPU cache cleared"}