import aiohttp
from typing import List
import os
import tiktoken

class Embeddings:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if not self.api_key:
            raise ValueError(f"{provider.upper()}_API_KEY environment variable not set")
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8000 

    async def embed_async(self, text: str) -> List[float]:
        chunks = self._chunk_text(text)
        embeddings = await self.embed_batch_async(chunks)
        return self._average_embeddings(embeddings)

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for text in texts:
            chunks = self._chunk_text(text)
            chunk_embeddings = await self._openai_embed_async(chunks)
            all_embeddings.append(self._average_embeddings(chunk_embeddings))
        return all_embeddings

    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk = self.tokenizer.decode(tokens[i:i + self.max_tokens])
            chunks.append(chunk)
        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        if not embeddings:
            return []
        avg_embedding = [sum(e) / len(embeddings) for e in zip(*embeddings)]
        return avg_embedding

    async def _openai_embed_async(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": texts,
                    "model": self.model
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"OpenAI API request failed with status {response.status}: {await response.text()}")
                data = await response.json()
                return [item['embedding'] for item in data['data']]