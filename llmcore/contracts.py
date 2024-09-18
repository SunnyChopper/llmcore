from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class LLMResponse:
    content: str
    raw_response: Dict[str, Any]

@dataclass
class LLMStreamResponse:
    content: str
    is_finished: bool

@dataclass
class ConversationTurn:
    role: str
    content: str

@dataclass
class Conversation:
    history: List[ConversationTurn]