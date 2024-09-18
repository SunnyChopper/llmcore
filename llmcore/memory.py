from typing import List, Dict, Any, Optional
from scipy import spatial
import numpy as np

from llmcore.vector_databases.vector_database_base import VectorDatabase
from llmcore.vector_databases.pinecone_database import PineconeDatabase
from llmcore.vector_databases.chroma_database import ChromaDatabase
from llmcore.core import LLMConfig
from llmcore.logger import log, setup_logger

logger = setup_logger(__name__)

def get_vector_database(config: LLMConfig) -> Optional[VectorDatabase]:
    if not config.vector_db_provider:
        return None
    provider = config.vector_db_provider.lower()
    if provider == "pinecone":
        return PineconeDatabase(endpoint=config.vector_db_endpoint, api_key=config.vector_db_api_key)
    elif provider == "chromadb":
        return ChromaDatabase(endpoint=config.vector_db_endpoint)
    # Add more providers here as needed
    else:
        raise ValueError(f"Unsupported vector database provider: {config.vector_db_provider}")

class MemoryManager:
    def __init__(self, config: LLMConfig, capacity: int = 32000):
        self.capacity = capacity
        self.memories: List[Dict[str, Any]] = []
        self.vector_db = get_vector_database(config)

    async def add_memory(self, memory: Dict[str, Any]):
        # Ensure 'vector' is a list of floats
        if not isinstance(memory.get('vector', []), list):
            raise ValueError("Memory 'vector' must be a list of floats.")
        if not all(isinstance(x, (int, float)) for x in memory['vector']):
            raise ValueError("All elements in memory 'vector' must be integers or floats.")
        
        if len(self.memories) >= self.capacity:
            self.memories.pop(0)
        self.memories.append(memory)
        
        if self.vector_db:
            try:
                await self.vector_db.add_vector(memory['vector'], {"content": memory['content']})
            except KeyError as e:
                raise ValueError(f"Memory dict is missing required key: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to add vector to database: {str(e)}") from e

    async def get_relevant_memories(self, query_vector: List[float], k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.vector_db:
            search_results = await self.vector_db.search_vectors(query_vector, top_k=k)
            return [{"content": res['metadata']['content'], "score": res['score']} for res in search_results.get('results', [])]
        else:
            try:
                # Ensure vectors are lists of floats
                similarities = [
                    self._calculate_similarity(query_vector, mem.get('vector', []))
                    for mem in self.memories
                    if isinstance(mem.get('vector', []), list) and all(isinstance(x, (int, float)) for x in mem['vector'])
                ]
                sorted_indices = np.argsort(similarities)[::-1]
                return [
                    {"content": self.memories[i].get('content', ''), "score": similarities[i]}
                    for i in sorted_indices[:k] if similarities[i] >= threshold
                ]
            except Exception as e:
                log(logger, "ERROR", f"Error retrieving relevant memories: {str(e)}")
                return []

    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        # Use cosine similarity for better performance and numerical stability
        return 1 - spatial.distance.cosine(v1, v2)

    def clear(self):
        self.memories.clear()
        if self.vector_db:
            # Implement vector DB clear if supported
            pass