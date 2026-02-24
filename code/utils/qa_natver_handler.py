import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple, Union

class QANatVerHandler:
    """
    Class that handles connections to QA-NatVer.

    Expected server endpoint:
        POST {base_url}/infer
        Body: {"items": [{"id": int, "claim": str, "evidence": [str, ...]}, ...]}
        Returns: {"results": [{"id": int, "verdict": {...}}, ...]}
    
    Internal States:
    - base_url: base URL of QA-NatVer server (e.g. http://127.0.0.1:8008)
    - log_stats: whether to log statistics of operation or not.
    """
    def __init__(
            self,
            base_url: str = "http://127.0.0.1:8008",
            timeout_s: Tuple[float, Optional[float]] = (3.05, 480.0),
            log_stats: bool = False,
            default_headers: Optional[Dict[str, str]] = None):

        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.log_stats = log_stats
        
        self.session = requests.Session()
        self.session.headers.update(default_headers or {"Content-Type": "application/json"})

    def health(self) -> Dict[str, Any]:
        """GET /health"""
        url = f"{self.base_url}/health"
        r = self.session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def unload(self) -> Dict[str, Any]:
        """POST /unload"""
        url = f"{self.base_url}/unload"
        r = self.session.post(url, json={}, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def infer(
            self,
            claims: List[str],
            evidences: List[List[str]],
            ids: Optional[List[int]] = None) -> Tuple[Dict[str, Any], float]:
        """Queries QA-NatVer server with a batch.
        
        Args:
            claims (List[str]): list of claim strings
            evidences (List[List[str]]): list of evidence lists
            ids (Optional[List[int]]): optional list of ids. If none uses 0...N-1

        Returns: 
            response_json (Dict): response given by QA-NatVer
                Example:
                    {
                        "results": [
                        {"id": 1, "verdict": {"pred_label": "...", ...}},
                        ...
                        ]
                    }
            total_time_s (float): time taken to process queries
        """

        assert len(claims) == len(evidences), "ERROR: claims and evidences must be same length"
        if ids is None:
            ids = list(range(len(claims)))
        else:
            assert len(ids) == len(claims), "ERROR: ids must be same length as claims"

        items = [{
            "id": int(ids[i]),
            "claim": str(claims[i]),
            "evidence": evidences[i],
        } for i in range(len(claims))]
        
        payload = {
            "items": items
        }
        url = f"{self.base_url}/infer"

        self.payload = json.dumps(payload, indent=4)

        if self.log_stats:
            print(f"[QANatVerHandler] POST {url}")
            print(f"[QANatVerHandler] Payload preview (first item): {items[0] if items else None}")

        start_time = time.time()
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_s)
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()

            total_time = time.time() - start_time

            if self.log_stats:
                print(f"Full response received {total_time:.2f} seconds after request")
                preview = data.get("results", [])[:1]
                print(f"[QANatVerHandler] Response preview: {json.dumps(preview, indent=2)[:1000]}")

            return data, total_time
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"ERROR: {e}", time.time() - start_time
        

    def infer_items(
            self,
            items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """Queries QA-NatVer server with a batch, already in correct format.
        
        Args:
            items (): items to directly send to QA-NatVer

        Returns: 
            response_json (Dict): response given by QA-NatVer
                Example:
                    {
                        "results": [
                        {"id": 1, "verdict": {"pred_label": "...", ...}},
                        ...
                        ]
                    }
            total_time_s (float): time taken to process queries
        """
        payload = {
            "items": items
        }
        url = f"{self.base_url}/infer"

        self.payload = json.dumps(payload, indent=4)

        if self.log_stats:
            print(f"[QANatVerHandler] POST {url}")
            print(f"[QANatVerHandler] Payload preview (first item): {items[0] if items else None}")

        start_time = time.time()
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_s)
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()

            total_time = time.time() - start_time

            if self.log_stats:
                print(f"Full response received {total_time:.2f} seconds after request")
                preview = data.get("results", [])[:1]
                print(f"[QANatVerHandler] Response preview: {json.dumps(preview, indent=2)[:1000]}")

            return data, total_time
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"ERROR: {e}", time.time() - start_time