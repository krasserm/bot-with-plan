from typing import Any, Dict, List, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from gba.grammar import schema_to_grammar


class LlamaCppClient(LLM):
    url: str
    n_predict: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64

    @property
    def _llm_type(self) -> str:
        return "llama-cpp"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            schema: Optional[Dict[str, Any]] = None,
            prop_order: Optional[List[str]] = (),
            prompt_ext: Optional[str] = None,
            **kwargs: Any,
    ) -> str:

        if prompt_ext is not None:
            prompt = prompt + prompt_ext

        payload = {
            "prompt": prompt,
            "stop": stop,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "stream": False,
        } | kwargs

        if schema is not None:
            payload["grammar"] = schema_to_grammar(schema, prop_order=prop_order)

        resp = requests.post(url=self.url, headers={"Content-Type": "application/json"}, json=payload)
        resp_json = resp.json()

        return resp_json["content"].strip()
