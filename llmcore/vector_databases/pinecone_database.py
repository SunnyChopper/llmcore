from typing import List, Dict, Any
import aiohttp
import os

from llmcore.vector_databases.vector_database_base import VectorDatabase

class PineconeDatabase(VectorDatabase):
    def __init__(self, endpoint: str, api_key: str, index_name: str = "llmcore-index"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.index_name = index_name
        self.headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

    async def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        data = {
            "vectors": [
                {
                    "id": metadata.get("id", os.urandom(16).hex()),
                    "values": vector,
                    "metadata": metadata
                }
            ]
        }
        url = f"{self.endpoint}/vectors/upsert"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Failed to add vector to Pinecone: {error}")

    async def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        data = {
            "vector": query_vector,
            "topK": top_k,
            "includeMetadata": True
        }
        url = f"{self.endpoint}/query"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Failed to search vectors in Pinecone: {error}")
                return await response.json()