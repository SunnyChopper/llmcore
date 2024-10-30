from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from asyncio import Queue, sleep
import threading
import asyncio
import aiohttp
import hashlib
import pickle
import ast
import os

# LLMCore
from llmcore.core import LLM, LLMConfig
from llmcore.utils import cosine_similarity
from llmcore.embeddings import Embeddings
from llmcore.prompt import PromptTemplate


@dataclass
class CodeSnippet:
    file_path: str
    content: str
    start_line: int
    end_line: int
    name: str = None
    snippet_type: str = None
    docstring: str = None
    embedding: List[float] = None
    relevance_score: float = 0.0

class CodebaseEmbeddings:
    CACHE_FILE = "codebase_embeddings.pkl"

    def __init__(self, embeddings: Embeddings, llm: Optional[LLM] = None):
        self.embeddings = embeddings
        self.snippets: List[CodeSnippet] = []
        self.chunk_embeddings: Dict[str, List[List[float]]] = defaultdict(list)
        self.llm = llm if llm else LLM(provider="google", model="gemini-1.5-flash", config = LLMConfig(temperature=0, max_tokens=2056, top_p=1))
        self.file_hashes: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "rb") as f:
                    cache_data = pickle.load(f)
                    self.snippets = cache_data.get("snippets", [])
                    self.file_hashes = cache_data.get("file_hashes", {})
            except Exception as e:
                print(f"Error loading cache: {e}")  # Replaced console.print with print

    def _save_cache(self):
        try:
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump({
                    "snippets": self.snippets,
                    "file_hashes": self.file_hashes
                }, f)
        except Exception as e:
            print(f"Error saving cache: {e}")  # Replaced console.print with print

    def _compute_file_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def build_embeddings(self, codebase_path: str, batch_size: int = 10) -> None:
        """
        Build embeddings for the given codebase path.

        This method walks through the codebase, processes each file, and generates
        embeddings for relevant code snippets (functions, classes, etc.) in parallel.

        Args:
            codebase_path (str): The path to the codebase directory.
            batch_size (int): Number of files to process in each batch.
        """
        async def process_file(file_path: str):
            try:
                with open(file_path, "r") as f:
                    content = f.read()
            except Exception as e:
                return None

            file_hash = self._compute_file_hash(content)
            cached_hash = self.file_hashes.get(file_path)

            if cached_hash == file_hash:
                return None  # File hasn't changed, skip processing

            # If one of restricted file extensions, skip processing
            if file_path.endswith((".pyc", ".pyo", ".pyd", ".pyw", ".pyz", ".pyzw")):
                return None
            
            try:
                if file_path.endswith(".py"):
                    snippets = self._extract_python_snippets(file_path, content)
                else:
                    snippets = await self._extract_generic_snippets(file_path, content)
            except Exception as e:
                print(f"Error extracting snippets from {file_path}: {e}")  # Replaced console.print with print
                return None

            if not snippets:
                return None
            
            try:
                embed_tasks = [self.embeddings.embed_async(snippet.content) for snippet in snippets]
                embeddings = await asyncio.gather(*embed_tasks)
            except Exception as e:
                print(f"Error generating embeddings for {file_path}: {e}")  # Replaced console.print with print
                return None

            try:
                for snippet, embedding in zip(snippets, embeddings):
                    snippet.embedding = embedding
            except Exception as e:
                print(f"Error assigning embeddings to snippets for {file_path}: {e}")  # Replaced console.print with print
                return None

            return file_path, snippets, file_hash

        def get_all_files(path):
            all_files = []
            for root, _, files in os.walk(path):
                all_files.extend([os.path.join(root, file) for file in files])
            return all_files

        # Use ThreadPoolExecutor to get all files without blocking
        with ThreadPoolExecutor() as executor:
            all_files = await asyncio.get_event_loop().run_in_executor(executor, get_all_files, codebase_path)

        total_files = len(all_files)

        # Set up interrupt handling
        stop_event = threading.Event()
        
        def interrupt_handler():
            stop_event.set()
            print("\nInterrupt received. Stopping gracefully...")  # Replaced console.print with print

        # Start a separate thread to handle keyboard interrupt
        def interrupt_watcher():
            try:
                while not stop_event.is_set():
                    stop_event.wait(1)
            except KeyboardInterrupt:
                interrupt_handler()

        watcher_thread = threading.Thread(target=interrupt_watcher)
        watcher_thread.daemon = True
        watcher_thread.start()

        request_queue = Queue()
        response_queue = Queue()

        # Populate the request queue with all files
        for file in all_files:
            request_queue.put_nowait(file)

        # Simple progress tracking
        processed_files = 0

        async def process_batch():
            nonlocal processed_files
            while not request_queue.empty():
                batch = []
                for _ in range(batch_size):
                    if request_queue.empty():
                        break
                    batch.append(request_queue.get_nowait())

                retry_count = 0
                while retry_count < 3:
                    try:
                        results = await asyncio.gather(*[process_file(file) for file in batch])
                        for result in results:
                            if result:
                                file_path, snippets, file_hash = result
                                self.snippets = [s for s in self.snippets if s.file_path != file_path]
                                self.snippets.extend(snippets)
                                self.file_hashes[file_path] = file_hash
                        processed_files += len(batch)
                        print(f"Processed {processed_files}/{total_files} files.")  # Progress update
                    except aiohttp.ClientResponseError as e:
                        if e.status == 429:
                            retry_count += 1
                            wait_time = 1.5 ** retry_count  # Exponential backoff
                            print(f"Rate limit hit. Retrying after {wait_time} seconds...")  # Replaced console.print with print
                            await sleep(wait_time)
                        else:
                            print(f"Error processing batch: {str(e)}")  # Replaced console.print with print
                            raise e     # Re-raise the exception to stop the processing
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")  # Replaced console.print with print
                        break
                
                # Add a small delay between batches
                await sleep(0.5)

        try:
            workers = [asyncio.create_task(process_batch()) for _ in range(5)]
            await asyncio.gather(*workers)

            if not stop_event.is_set():
                self._save_cache()
                print("Embedding build complete!")  # Replaced console.print with print
            else:
                print("Embedding build interrupted. Partial results saved.")  # Replaced console.print with print
        
        finally:
            stop_event.set()  # Ensure the watcher thread stops
            watcher_thread.join()  # Wait for the watcher thread to finish
            self._save_cache()

    def _extract_python_snippets(self, file_path: str, content: str) -> List[CodeSnippet]:
        """
        Extract relevant code snippets from a Python file.

        This method uses the ast module to parse Python code and extract
        functions, classes, and methods as separate snippets.

        Args:
            file_path (str): The path to the Python file.
            content (str): The content of the Python file.

        Returns:
            List[CodeSnippet]: A list of extracted code snippets.
        """
        snippets = []
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                snippet_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
                snippet_content = ast.get_source_segment(content, node)
                docstring = ast.get_docstring(node)
                
                if not docstring:
                    docstring = self._generate_docstring(snippet_type, node.name, snippet_content)
                
                snippets.append(CodeSnippet(
                    file_path=file_path,
                    content=snippet_content,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    snippet_type=snippet_type,
                    name=node.name,
                    docstring=docstring
                ))
        return snippets

    async def _extract_generic_snippets(self, file_path: str, content: str) -> List[CodeSnippet]:
        """
        Extract relevant code snippets from a non-Python file using an LLM.

        This method uses a fast LLM to identify and extract relevant code snippets
        from files that are not Python (e.g., JavaScript, Java, etc.).

        Args:
            file_path (str): The path to the file.
        - content (str): The content of the file.

        Returns:
            List[CodeSnippet]: A list of extracted code snippets.
        """
        @dataclass
        class ExtractedSnippet:
            snippet_type: str
            name: str
            start_line: int
            end_line: int
            docstring: str

        prompt_template = PromptTemplate(
            "Analyze the following code and extract relevant snippets (functions, classes, methods, etc.).\n"
            "For each snippet, provide:\n"
            "1. The snippet type (function, class, method, etc.)\n"
            "2. The snippet name\n"
            "3. The start and end line numbers\n"
            "4. A brief docstring describing the snippet's purpose\n\n"
            "Code:\n```\n{{code}}\n```\n\n"
            "Provide the output in JSON format.",
            required_params={"code": str},
            output_json_structure={
                "snippets": List[ExtractedSnippet]
            }
        )

        prompt = prompt_template.create_prompt(code=content)
        result = await self.llm.send_input_async(prompt, parse_json=True)

        snippets = []
        for snippet_data in result.get("snippets", []):
            snippets.append(CodeSnippet(
                file_path=file_path,
                content="\n".join(content.split("\n")[snippet_data["start_line"]-1:snippet_data["end_line"]]),
                start_line=snippet_data["start_line"],
                end_line=snippet_data["end_line"],
                snippet_type=snippet_data["type"],
                name=snippet_data["name"],
                docstring=snippet_data["docstring"]
            ))
        return snippets

    def _generate_docstring(self, snippet_type: str, name: str, content: str) -> str:
        """
        Generate a docstring for a code snippet using an LLM.

        Args:
            snippet_type (str): The type of the snippet (function, class, etc.).
            name (str): The name of the snippet.
            content (str): The content of the snippet.

        Returns:
            str: A generated docstring for the snippet.
        """
        prompt_template = PromptTemplate(
            "Generate a brief docstring for the following {{snippet_type}} named '{{name}}':\n\n"
            "```\n{{content}}\n```\n\n"
            "Provide only the docstring, without quotes.",
            required_params={"snippet_type": str, "name": str, "content": str}
        )
        prompt = prompt_template.create_prompt(snippet_type=snippet_type, name=name, content=content)
        return self.llm.send_input(prompt)

    async def get_relevant_snippets(self, query: str, top_k: int = 5, snippet_type: str = "function") -> List[CodeSnippet]:
        """
        Retrieve the most relevant code snippets of a specified type for a given query.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            snippet_type (str): The type of snippets to return (e.g., "function", "class", "method").

        Returns:
            List[CodeSnippet]: A list of the most relevant code snippets of the specified type.
        """
        query_embedding = await self.embeddings.embed_async(query)
    
        # Filter out snippets with empty embeddings and match the specified type
        valid_snippets = [
            snippet for snippet in self.snippets 
            if snippet.embedding and len(snippet.embedding) > 0 and snippet.snippet_type == snippet_type
        ]
        
        if len(valid_snippets) == 0:
            print(f"No relevant {snippet_type} snippets found for the given query.")  # Replaced console.print with print
            return []

        scored_snippets = [
            (snippet, cosine_similarity(query_embedding, snippet.embedding))
            for snippet in valid_snippets
        ]
        
        sorted_snippets = sorted(scored_snippets, key=lambda x: x[1], reverse=True)
        
        for snippet, score in sorted_snippets[:top_k]:
            snippet.relevance_score = score
        
        return [snippet for snippet, _ in sorted_snippets[:top_k]]

    def get_complex_functions(self, threshold: float = 0.7) -> List[CodeSnippet]:
        """
        Identify complex functions in the codebase based on a complexity threshold.

        Args:
            threshold (float): The complexity threshold (0.0 to 1.0).

        Returns:
            List[CodeSnippet]: A list of complex code snippets.
        """
        complex_snippets = []
        for snippet in self.snippets:
            if snippet.snippet_type == "function":
                complexity = self._calculate_complexity(snippet.content)
                if complexity > threshold:
                    complex_snippets.append(snippet)
        return complex_snippets

    def _calculate_complexity(self, content: str) -> float:
        """
        Calculate the complexity of a code snippet.

        This is a simple implementation that can be expanded with more sophisticated metrics.

        Args:
            content (str): The content of the code snippet.

        Returns:
            float: A complexity score between 0.0 and 1.0.
        """
        lines = content.split("\n")
        loc = len(lines)
        cyclomatic_complexity = content.count("if ") + content.count("for ") + content.count("while ") + 1
        return min(1.0, (cyclomatic_complexity / 10) * (loc / 50))
