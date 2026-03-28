import os
from openai import OpenAI
from typing import Dict, List, Optional, Any

# ─── API CONFIGURATION ───
GPT_OSS_KEY = "sk-nc-O03Rv5xDzQ__ST2lmMsHK2L1icMUBlkpPQgB78SDn1E"
DEEPSEEK_KEY = "sk-nc-d3gyflOlE4sB_g1xo_nxGrdImjN6L6k4quTuXvOQhvk"

# Official NeevCloud endpoint for DeepSeek R1
ENDPOINTS = [
    "https://inference.ai.neevcloud.com/v1",
]

class LLMClient:
    """Centralized LLM client with fallback logic."""
    
    def __init__(self, model_type: str = "deepseek"):
        self.model_type = model_type
        if model_type == "gpt-oss":
            self.key = GPT_OSS_KEY
            self.model = "gpt-oss-120b"
        else:
            self.key = DEEPSEEK_KEY
            self.model = "deepseek-v3-2"
        
        # We'll initialize the client with the first endpoint, but can re-init on failure
        self.current_endpoint_idx = 0
        self.client = OpenAI(api_key=self.key, base_url=ENDPOINTS[0], timeout=120.0)

    def generate_content(self, prompt: str, system_prompt: str = "You are a professional assistant.", json_mode: bool = False) -> str:
        """Completion wrapper with endpoint fallback."""
        last_err = None
        
        for i in range(len(ENDPOINTS)):
            try:
                base_url = ENDPOINTS[(self.current_endpoint_idx + i) % len(ENDPOINTS)]
                self.client.base_url = base_url
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # If successful, remember this endpoint
                self.current_endpoint_idx = (self.current_endpoint_idx + i) % len(ENDPOINTS)
                content = response.choices[0].message.content
                
                # DeepSeek R1 wraps output in <think>...</think> tags — strip them
                if content and "<think>" in content:
                    import re
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                
                # Strip markdown json blocks if present
                if content and content.startswith("```json"):
                    content = content.replace("```json\n", "").replace("\n```", "").strip()
                if content and content.startswith("```"):
                    content = content.replace("```\n", "").replace("\n```", "").strip()
                
                return content
                
            except Exception as e:
                last_err = e
                print(f"[LLMClient] Attempt with {base_url} failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"LLM Final Error ({self.model}): {last_err}")
        raise last_err

    def get_client(self):
        return self.client
