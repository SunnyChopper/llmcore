from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict
import os

# LLMCore
from llmcore.utils import cosine_similarity
from llmcore.embeddings import Embeddings
from llmcore.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class CodeSnippet:
    file_path: str
    content: str
    start_line: int
    end_line: int
    function_name: str = None
    class_name: str = None
    relevance_score: float = 0.0

class CodebaseEmbeddings:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.chunk_size = 10
        self.stride = 5
        self.chunk_embeddings: Dict[str, List[List[float]]] = defaultdict(list)

    def build_embeddings(self, codebase_path: str) -> None:
        for root, _, files in os.walk(codebase_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        content = f.read()
                    chunks = sliding_window_chunker(content, window_size=self.chunk_size, stride=self.stride)
                    for chunk in chunks:
                        embedding = self.embeddings.embed_async(chunk)  # Ensure this is awaited in async context
                        self.chunk_embeddings[file_path].append(embedding)

    async def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[CodeSnippet]:
        query_vector = await self.embeddings.embed_async(query)
        relevant_chunks = []
        for file_path, embeddings_list in self.chunk_embeddings.items():
            for idx, chunk_vector in enumerate(embeddings_list):
                similarity = cosine_similarity(query_vector, chunk_vector)
                if similarity > 0.7:  # Threshold can be adjusted
                    content = sliding_window_chunker(open(file_path).read())[idx]
                    start_line = get_start_line(open(file_path).read(), content)
                    end_line = get_end_line(open(file_path).read(), content)
                    function_name = extract_function_name(content)
                    class_name = extract_class_name(content)
                    snippet = CodeSnippet(
                        file_path=file_path,
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                        function_name=function_name,
                        class_name=class_name,
                        relevance_score=similarity
                    )
                    relevant_chunks.append(snippet)
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return relevant_chunks[:top_k]

def sliding_window_chunker(content: str, window_size: int = 10, stride: int = 5) -> List[str]:
    lines = content.split('\n')
    return ['\n'.join(lines[i:i+window_size]) for i in range(0, len(lines) - window_size + 1, stride)]

def get_start_line(content: str, chunk: str) -> int:
    return content.count('\n', 0, content.index(chunk)) + 1

def get_end_line(content: str, chunk: str) -> int:
    return get_start_line(content, chunk) + chunk.count('\n')

def extract_function_name(chunk: str) -> str:
    import re
    match = re.search(r'def\s+(\w+)', chunk)
    return match.group(1) if match else None

def extract_class_name(chunk: str) -> str:
    import re
    match = re.search(r'class\s+(\w+)', chunk)
    return match.group(1) if match else None