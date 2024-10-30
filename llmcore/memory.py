from typing import List, Dict, Any, Optional, Union, Sequence
from math import sqrt

from llmcore.vector_databases.vector_database_base import VectorDatabase
from llmcore.vector_databases.pinecone_database import PineconeDatabase
from llmcore.vector_databases.chroma_database import ChromaDatabase
from llmcore.core import LLMConfig

class Vector:
    def __init__(self, data: Union[Sequence[float], 'Vector']):
        if isinstance(data, Vector):
            self.data = data.data
        elif not isinstance(data, (list, tuple)):
            raise TypeError("Vector data must be a list of floats or a Vector.")
        else:
            try:
                self.data = [float(x) for x in data]
            except (ValueError, TypeError):
                raise TypeError("Vector data must contain only numeric values")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def dot(self, other: 'Vector') -> float:
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def norm(self) -> float:
        return sqrt(sum(x * x for x in self.data))
    
    def tolist(self) -> List[float]:
        return self.data
    
    @property
    def shape(self) -> tuple:
        return (len(self.data),)

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
        self.vector_dim = None

    async def add_memory(self, memory: Dict[str, Any]):
        vector_data = memory.get('vector')
        if not isinstance(vector_data, (list, Vector)):
            raise ValueError("Memory 'vector' must be a list of floats or a Vector.")
        
        vector = Vector(vector_data)
        
        if self.vector_dim is None:
            self.vector_dim = len(vector)
        elif len(vector) != self.vector_dim:
            raise ValueError(f"Memory 'vector' must have dimension {self.vector_dim}.")
        
        memory['vector'] = vector
        
        if len(self.memories) >= self.capacity:
            self.memories.pop(0)
        self.memories.append(memory)
        
        if self.vector_db:
            try:
                await self.vector_db.add_vector(vector.tolist(), {"content": memory['content']})
            except KeyError as e:
                raise ValueError(f"Memory dict is missing required key: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to add vector to database: {str(e)}") from e

    async def get_relevant_memories(self, query_vector: List[float], k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        from llmcore.core import RelevantMemory
        
        query_vec = Vector(query_vector)
        if self.vector_dim is not None and len(query_vec) != self.vector_dim:
            raise ValueError(f"Query vector must have dimension {self.vector_dim}.")

        if self.vector_db:
            results = await self.vector_db.search_vectors(query_vec.tolist(), k)
            return [RelevantMemory(content=result['content'], score=result['score']) 
                    for result in results if result['score'] >= threshold]
        else:
            def convert_to_vector(vec):
                if isinstance(vec, (list, Vector)):
                    return Vector(vec)
                elif isinstance(vec, str):
                    try:
                        return Vector([float(x) for x in vec.strip('[]').split(',')])
                    except ValueError:
                        print(f"Error converting string to vector: {vec[:100]}...")
                        return None
                else:
                    print(f"Unexpected vector type: {type(vec)}")
                    return None

            try:
                valid_memories = []
                for mem in self.memories:
                    converted_vector = convert_to_vector(mem.get('vector'))
                    if converted_vector is not None:
                        mem['vector'] = converted_vector
                        valid_memories.append(mem)
            except Exception as e:
                print(f"Error processing memories: {str(e)}")
                return []

            try:
                similarities = []
                for mem in valid_memories:
                    try:
                        sim = self._calculate_similarity(query_vec, mem['vector'])
                        similarities.append(sim)
                    except Exception as e:
                        print(f"Error calculating similarity: {str(e)}")
            except Exception as e:
                print(f"Error calculating similarities: {str(e)}")
                return []

            try:
                # Sort indices in descending order of similarity
                sorted_pairs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
                results = [
                    RelevantMemory(content=valid_memories[idx].get('content', ''), score=score)
                    for idx, score in sorted_pairs[:k] if score >= threshold
                ]
                return results
            except Exception as e:
                print(f"Error sorting and filtering results: {str(e)}")
                return []

    def _calculate_similarity(self, vector1: Vector, vector2: Vector) -> float:
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vector1 (Vector): The first vector.
            vector2 (Vector): The second vector.

        Returns:
            float: The cosine similarity between vector1 and vector2.

        Raises:
            ValueError: If either vector has zero magnitude.
        """
        try:
            dot_product = vector1.dot(vector2)
            norm1 = vector1.norm()
            norm2 = vector2.norm()

            if norm1 == 0.0 or norm2 == 0.0:
                raise ValueError("One or both vectors have zero magnitude, cannot compute cosine similarity.")

            cosine_similarity = dot_product / (norm1 * norm2)
            return max(min(cosine_similarity, 1.0), -1.0)
        except Exception as e:
            print(f"Error in _calculate_similarity: {str(e)}")
            raise

    def clear(self):
        self.memories.clear()
        if self.vector_db:
            # Implement vector DB clear if supported
            pass