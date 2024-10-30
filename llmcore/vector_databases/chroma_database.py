from typing import List, Dict, Any
import aiohttp
import os

from llmcore.vector_databases.vector_database_base import VectorDatabase

class ChromaDatabase(VectorDatabase):
    def __init__(self, endpoint: str = "http://localhost:8000", collection_name: str = "llmcore-collection"):
        self.endpoint = endpoint
        self.collection_name = collection_name
        self.headers = {
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
        url = f"{self.endpoint}/collections/{self.collection_name}/vectors"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                if response.status not in (200, 201):
                    error = await response.text()
                    raise Exception(f"Failed to add vector to ChromaDB: {error}")

    async def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        data = {
            "query_vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        url = f"{self.endpoint}/collections/{self.collection_name}/search"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Failed to search vectors in ChromaDB: {error}")
                return await response.json()