from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDatabase(ABC):
    @abstractmethod
    async def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass