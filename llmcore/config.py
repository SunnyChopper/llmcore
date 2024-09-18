from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class LLMConfig:
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    response_format: Optional[Dict] = None
    json_response: bool = False
    json_instruction: Optional[str] = None
    vector_db_provider: Optional[str] = None  # Added vector DB provider
    vector_db_endpoint: Optional[str] = None
    vector_db_api_key: Optional[str] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}