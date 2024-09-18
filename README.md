# LLMCore

LLMCore is a powerful Python library for working with Large Language Models (LLMs). It provides a flexible and extensible interface for interacting with various LLM providers and building complex workflows.

## Features

- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and Google (Gemini).
- **Asynchronous and Synchronous API Calls**: Flexibility in how you interact with LLMs.
- **Prompt Templating with Parameter Validation**: Create reusable and validated prompts.
- **Automatic JSON Parsing of LLM Responses**: Easily handle structured responses.
- **LLM Call Chaining for Complex Workflows**: Build multi-step interactions.
- **Configurable LLM Parameters**: Customize temperature, max tokens, top-p, and more.
- **Memory Management**: Maintain conversational context across interactions.
- **Embeddings for Semantic Searches and Codebase Analysis**: Enhance search and analysis capabilities.
- **Comprehensive Logging**: Colored and leveled logging for better debugging and monitoring.

## Installation

You can install LLMCore using pip:

```bash
pip install llmcore
```

## Configuration

Before using LLMCore, you need to set up API keys for the LLM providers you want to use. Set the following environment variables:

- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic
- `GEMINI_API_KEY` for Google Gemini

## Usage

### Initializing an LLM
To initialize an LLM, use the `LLM` class and specify the provider and model you want to use. For example:

```python
from llmcore.core import LLM

llm = LLM(provider="openai", model="gpt-4o")
```

#### Setting Configuration

You can also set configuration options for the LLM, such as temperature, max_tokens, and top_p:

```python
from llmcore.core import LLM, LLMConfig

config = LLMConfig(
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

llm = LLM(provider="openai", model="gpt-4o", config=config)
```

The full list of configuration options is in the table below:

| Option            | Type    | Description                                               |
| ----------------- | ------- | --------------------------------------------------------- |
| temperature       | float   | Controls the randomness of the LLM output.                |
| max_tokens        | int     | Limits the number of tokens in the LLM output.            |
| top_p             | float   | Controls the diversity of the LLM output.                 |
| response_format   | dict    | Specifies the format of the LLM response.                 |
| json_response     | bool    | Whether to parse the LLM output as JSON.                  |
| json_instruction  | string  | Instruction to ensure JSON formatting in LLM responses.    |

### Send a Simple Prompt

You can send a simple prompt to the LLM using the `send_input` method. For example:

```python
from llmcore.core import LLM

openai_llm = LLM(provider="openai", model="gpt-4o")
response = openai_llm.send_input("Explain the concept of recursion in programming.")
print(response)
```

### Streaming Output

Stream the output of the LLM using the `stream_input` method. For example:

```python
from llmcore.core import LLM

openai_llm = LLM(provider="openai", model="gpt-4o")
for chunk in openai_llm.stream_input("Write a short story about a robot learning to paint."):
    print(chunk)
```

### Using Prompt Templates

Create prompt templates to reuse prompts with dynamic content. For example:

```python
from llmcore.core import LLM
from llmcore.prompt import PromptTemplate

code_review_template = PromptTemplate(
    "Review the following {{language}} code:\n\n```{{language}}\n{{code}}\n```\n\nProvide feedback on code quality, potential bugs, and suggestions for improvement.",
    required_params={"language": str, "code": str}
)

# Use the template to create a prompt
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

prompt = code_review_template.create_prompt(language="python", code=python_code)

# Send the prompt to the LLM
response = openai_llm.send_input(prompt)
print(response)
```

## Advanced Usage Examples

### Example 1: Summarizing and Translating Text Using LLM Chains

```python
from llmcore.core import LLM
from llmcore.chain import LLMChainBuilder
from llmcore.prompt import PromptTemplate

# Initialize the LLM
llm = LLM(provider="openai", model="gpt-4o")

# Build the LLM chain
summarize_and_translate = (LLMChainBuilder(llm)
    .add_step(
        template="Summarize the following text in 2-3 sentences:\n\n{{input_text}}",
        output_key="summary",
        required_params={"input_text": str},
        output_json_structure={"summary": str}
    )
    .add_step(
        template="Translate the following summary to French:\n\n{{summary}}",
        output_key="french_summary",
        required_params={"summary": str},
        output_json_structure={"french_summary": str}
    )
    .build()
)

# Execute the chain
result = summarize_and_translate.execute({
    "input_text": "OpenAI has developed a new model that surpasses previous benchmarks in natural language understanding."
})

print(f"Summary: {result['summary']}")
print(f"French Summary: {result['french_summary']}")
```

### Example 2: Managing Conversation Memory

```python
from llmcore.core import LLM, PromptTemplate
from llmcore.memory import MemoryManager
from llmcore.contracts import ConversationTurn

# Initialize the LLM with memory
memory_manager = MemoryManager(config=LLMConfig(), capacity=32000)
llm = LLM(provider="openai", model="gpt-4o", memory_manager=memory_manager)

# Define a prompt template
conversation_template = PromptTemplate(
    "Human: {{human_input}}\nAI:",
    required_params={"human_input": str},
    output_json_structure={"response": str}
)

# Start a conversation with memory
response = llm.send_input_with_memory(
    conversation_template.create_prompt(human_input="Hello, who are you?"),
    parse_json=True
)
print(response)

# Continue the conversation
response = llm.send_input_with_memory(
    conversation_template.create_prompt(human_input="Tell me a joke."),
    parse_json=True
)
print(response)
```

### Example 3: Embedding and Semantic Search

```python
import asyncio
from llmcore.embeddings import Embeddings, CodebaseEmbeddings
from llmcore.utils import cosine_similarity

async def main():
    # Initialize embeddings
    embeddings = Embeddings(provider="openai", model="text-embedding-3-small")

    # Initialize codebase embeddings
    codebase_embeddings = CodebaseEmbeddings(embeddings)

    # Embed a code file
    file_embedding = await codebase_embeddings.get_file_embedding("path/to/file.py")

    # The query is the user's input
    query = "How to implement a binary search in Python?"

    # Create a vector representation of the query
    query_embedding = await embeddings.embed_async(query)

    # Perform semantic search
    similarities = [cosine_similarity(query_embedding, emb) for emb in codebase_embeddings.file_embeddings.values()]
    top_matches = sorted(zip(codebase_embeddings.file_embeddings.keys(), similarities), key=lambda x: x[1], reverse=True)[:5]

    for file, score in top_matches:
        print(f"File: {file}, Similarity: {score}")

asyncio.run(main())
```

### Example 4: Prompt Templating with Parameter Validation

```python
from llmcore.core import LLM
from llmcore.prompt import PromptTemplate
from typing import List

# Define a prompt template for code review
code_review_template = PromptTemplate(
    "Review the following {{language}} code:\n\n```{{language}}\n{{code}}\n```\n\nProvide feedback on code quality, potential bugs, and suggestions for improvement.",
    required_params={"language": str, "code": str},
    output_json_structure={"feedback": str, "bugs": List[str], "suggestions": List[str]}
)

# Create a prompt
prompt = code_review_template.create_prompt(language="python", code="""
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
""")

# Initialize the LLM
llm = LLM(provider="openai", model="gpt-4o")

# Send the prompt and parse JSON response
response = llm.send_input(prompt, parse_json=True)
print(response["feedback"])
print(response["bugs"])
print(response["suggestions"])
```

### Example 5: Chaining Multiple LLM Calls with Memory

```python
from llmcore.core import LLM
from llmcore.chain import LLMChainBuilder
from llmcore.prompt import PromptTemplate
from llmcore.memory import MemoryManager

# Initialize the LLM
llm = LLM(provider="openai", model="gpt-4o")

# Define prompt templates
summarize_template = PromptTemplate(
    "Summarize the following article in 3 sentences:\n\n{{article}}",
    required_params={"article": str},
    output_json_structure={"summary": str}
)

analyze_sentiment_template = PromptTemplate(
    "Analyze the sentiment of the following summary:\n\n{{summary}}",
    required_params={"summary": str},
    output_json_structure={"sentiment": str}
)

# Build the LLM chain
summarize_and_analyze = (LLMChainBuilder(llm)
    .add_step(
        template=summarize_template.template,
        output_key="summary",
        required_params=summarize_template.required_params,
        output_json_structure=summarize_template.output_json_structure
    )
    .add_step(
        template=analyze_sentiment_template.template,
        output_key="sentiment",
        required_params=analyze_sentiment_template.required_params,
        output_json_structure=analyze_sentiment_template.output_json_structure
    )
    .build()
)

# Execute the chain with memory
result = summarize_and_analyze.execute({
    "article": "OpenAI has released a new model that significantly improves natural language understanding..."
})

print(f"Summary: {result['summary']}")
print(f"Sentiment: {result['sentiment']}")
```

## API Reference

### `llmcore/core.py`

#### Classes

- **LLMConfig**

  ```python
  @dataclass
  class LLMConfig:
      temperature: float = 0.7
      max_tokens: int = 150
      top_p: float = 1.0
      response_format: Optional[Dict] = None
      json_response: bool = False
      json_instruction: Optional[str] = None

      def to_dict(self) -> Dict:
          return {k: v for k, v in asdict(self).items() if v is not None}
  ```

  - **Attributes:**
    - `temperature`: Controls the randomness of the LLM output.
    - `max_tokens`: Limits the number of tokens in the LLM output.
    - `top_p`: Controls the diversity of the LLM output.
    - `response_format`: Specifies the format of the LLM response.
    - `json_response`: Whether to parse the LLM output as JSON.
    - `json_instruction`: Instruction to ensure JSON formatting in LLM responses.

- **LLMClientAdapter (Abstract Base Class)**

  ```python
  class LLMClientAdapter(ABC):
      @abstractmethod
      async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
          pass

      @abstractmethod
      async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
          pass
  ```

  - **Methods:**
    - `send_prompt`: Sends a prompt to the LLM and returns the response as a string.
    - `stream_prompt`: Sends a prompt to the LLM and yields response chunks asynchronously.

- **APIClientAdapter (Concrete Class)**

  Handles interaction with different LLM APIs.

  ```python
  class APIClientAdapter(LLMClientAdapter):
      ...
  ```

  - **Methods:**
    - `_make_request`: Makes HTTP requests to the LLM API with retry and error handling.
    - `_stream_request`: Handles streaming requests to the LLM API.
    - `_stream_<provider>`: Stream processing for specific LLM providers (e.g., OpenAI, Anthropic, Google Gemini).
    - `send_prompt`: Implements sending prompts for the specific provider.
    - `stream_prompt`: Implements streaming prompts for the specific provider.
  
- **OpenAIClientAdapter, AnthropicClientAdapter, GoogleGeminiClientAdapter**

  Subclasses of `APIClientAdapter` tailored for specific LLM providers.

  ```python
  class OpenAIClientAdapter(APIClientAdapter):
      ...
  class AnthropicClientAdapter(APIClientAdapter):
      ...
  class GoogleGeminiClientAdapter(APIClientAdapter):
      ...
  ```

- **LLM**

  Main class to interact with LLM providers.

  ```python
  class LLM:
      ...
  ```

  - **Methods:**
    - `send_input`: Send a prompt and receive a response synchronously.
    - `stream_input`: Stream a prompt response synchronously.
    - `send_input_async`: Asynchronous version of `send_input`.
    - `stream_input_async`: Asynchronous version of `stream_input`.
    - `send_input_with_history`: Send a prompt with conversation history.
    - `send_input_with_memory`: Send a prompt with memory context.
    - `update_config`: Update LLM configuration.
    - Additional internal methods for JSON parsing, type validation, etc.

### `llmcore/chain.py`

#### Classes

- **LLMChainStep**

  Represents a single step in an LLM chain.

  ```python
  class LLMChainStep:
      def __init__(self, prompt_template: PromptTemplate, output_key: str, llm: Optional[LLM] = None):
          ...
  ```

- **LLMChain**

  Manages a sequence of `LLMChainStep` to execute complex workflows.

  ```python
  class LLMChain:
      def __init__(self, default_llm: LLM, steps: List[LLMChainStep], use_memory: bool = False):
          ...
      
      def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
          ...
      
      async def execute_async(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
          ...
      
      def stream(self, initial_input: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
          ...
      
      async def stream_async(self, initial_input: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
          ...
      
      # Additional internal methods for JSON extraction and input validation
  ```

- **LLMChainBuilder**

  Builder class to construct `LLMChain` instances.

  ```python
  class LLMChainBuilder:
      def __init__(self, default_llm: LLM = None, use_memory: bool = False):
          ...
      
      def add_step(self, template: str, output_key: str, required_params: Optional[Dict[str, type]] = None,
                   output_json_structure: Optional[Dict[str, Any]] = None,
                   llm: Optional[Union[LLM, Dict[str, Any]]] = None) -> 'LLMChainBuilder':
          ...
      
      def build(self) -> LLMChain:
          ...
      
      # Additional internal methods for placeholder extraction
  ```

### `llmcore/utils.py`

Utility functions used across the library.

```python
import numpy as np
from typing import List

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))
```

### `llmcore/memory.py`

Handles memory management for context-aware interactions with LLMs.

- **VectorDatabase**

  ```python
  class VectorDatabase:
      def __init__(self):
          self.memory = []
      
      def add_memory(self, vector):
          self.memory.append(vector)
      
      def search_memory(self, query_vector):
          # Implement search logic here
          pass
  ```

- **MemoryManager**

  ```python
  class MemoryManager:
      def __init__(self, capacity: int = 100):
          self.capacity = capacity
          self.memories: List[Dict[str, Any]] = []
      
      def add_memory(self, memory: Dict[str, Any]):
          if len(self.memories) >= self.capacity:
              self.memories.pop(0)
          self.memories.append(memory)
      
      def get_relevant_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
          # Simple similarity search implementation
          similarities = [self._calculate_similarity(query, mem['content']) for mem in self.memories]
          sorted_indices = np.argsort(similarities)[::-1]
          return [self.memories[i] for i in sorted_indices[:k]]
      
      def _calculate_similarity(self, query: str, memory_content: str) -> float:
          # Basic similarity calculation using word overlap
          query_words = set(query.lower().split())
          memory_words = set(memory_content.lower().split())
          return len(query_words.intersection(memory_words)) / len(query_words.union(memory_words))
      
      def clear(self):
          self.memories.clear()
  ```

### `llmcore/prompt.py`

Handles prompt templating with placeholders and input validation.

- **PromptTemplate**

  ```python
  class PromptTemplate:
      def __init__(self, template: str, required_params: Dict[str, type], output_json_structure: Optional[Dict[str, Any]] = None):
          self.template = template
          self.required_params = required_params
          self.output_json_structure = self._convert_types(output_json_structure) if output_json_structure else None
          self.placeholders = self._extract_placeholders()
      
      def _convert_types(self, structure: Dict[str, Any]) -> Dict[str, Any]:
          def convert(item):
              if isinstance(item, type):
                  return item.__name__
              elif isinstance(item, list):
                  return [convert(i) for i in item]
              elif isinstance(item, dict):
                  return {k: convert(v) for k, v in item.items()}
              elif hasattr(item, '__origin__'):  # Handles Union and generics
                  origin = item.__origin__
                  args = item.__args__
                  if origin is Union:
                      return f"Union[{', '.join(convert(arg) for arg in args)}]"
                  return f"{origin.__name__}[{', '.join(convert(arg) for arg in args)}]"
              else:
                  return str(item)
      
          return {k: convert(v) for k, v in structure.items()}
      
      def _extract_placeholders(self) -> set:
          return set(re.findall(r'{{(\w+)}}', self.template))
      
      def _validate_inputs(self, kwargs: Dict[str, Any]):
          for key, expected_type in self.required_params.items():
              if key not in kwargs:
                  raise ValueError(f"Missing required parameter: {key}")
              
              value = kwargs[key]
              origin_type = get_origin(expected_type) or expected_type
              
              if origin_type in (List, list):
                  if not isinstance(value, list):
                      raise TypeError(f"Parameter {key} should be a list")
                  if value and get_args(expected_type):
                      item_type = get_args(expected_type)[0]
                      if not all(isinstance(item, item_type) for item in value):
                          raise TypeError(f"All items in {key} should be of type {item_type}")
              elif origin_type in (Dict, dict):
                  if not isinstance(value, dict):
                      raise TypeError(f"Parameter {key} should be a dict")
                  if get_args(expected_type):
                      key_type, value_type = get_args(expected_type)
                      if not all(isinstance(k, key_type) and isinstance(v, value_type) for k, v in value.items()):
                          raise TypeError(f"All items in {key} should be of type {key_type}: {value_type}")
              elif not isinstance(value, origin_type):
                  raise TypeError(f"Parameter {key} should be of type {expected_type}")
      
      def create_prompt(self, **kwargs) -> 'Prompt':
          self._validate_inputs(kwargs)
          return Prompt(self, **kwargs)
  ```

- **Prompt**

  Represents an instantiated prompt with specific values.

  ```python
  class Prompt:
      def __init__(self, template: PromptTemplate, **kwargs):
          self.template = template
          self.values = kwargs
      
      def _sanitize_input(self, value: Any) -> str:
          sanitized = str(value)
          sanitized = re.sub(r'[<>]', '', sanitized)
          return sanitized
      
      def format(self) -> str:
          sanitized_kwargs = {}
          for k, v in self.values.items():
              if isinstance(v, dict):
                  sanitized_kwargs[k] = {sk: self._sanitize_input(sv) for sk, sv in v.items()}
              else:
                  sanitized_kwargs[k] = self._sanitize_input(v)
          
          def replace_placeholder(match):
              placeholder = match.group(1)
              start = match.start()
              end = match.end()
              
              # Check if the placeholder is within triple quotes
              triple_quote_before = self.template.template.rfind('"""', 0, start)
              triple_quote_after = self.template.template.find('"""', end)
              
              # Check if the placeholder is within backticks (for code examples)
              backtick_before = self.template.template.rfind('`', 0, start)
              backtick_after = self.template.template.find('`', end)
              
              # Check if the placeholder is within a Python-style string (single or double quotes)
              single_quote_before = self.template.template.rfind("'", 0, start)
              single_quote_after = self.template.template.find("'", end)
              double_quote_before = self.template.template.rfind('"', 0, start)
              double_quote_after = self.template.template.find('"', end)
              
              # If within any of these quote types, don't replace
              if (triple_quote_before != -1 and triple_quote_after != -1) or \
                 (backtick_before != -1 and backtick_after != -1) or \
                 (single_quote_before != -1 and single_quote_after != -1) or \
                 (double_quote_before != -1 and double_quote_after != -1):
                  return match.group(0)
              
              # Handle nested placeholders
              parts = placeholder.split('.')
              value = sanitized_kwargs
              for part in parts:
                  if isinstance(value, dict) and part in value:
                      value = value[part]
                  else:
                      return match.group(0)  # Placeholder not found, return original
              return str(value)
          
          formatted_prompt = re.sub(r'{{([\w.]+)}}', replace_placeholder, self.template.template)
          
          if self.template.output_json_structure:
              json_instruction = "\n\nPlease provide your response in the following JSON format and ensure that the JSON is properly enclosed within triple backticks. Do not include any other text or formatting. Be very careful with the formatting of the JSON, as it may break the rest of the application if not done correctly! Finally, ensure that the JSON produced is not the escaped version of the JSON structure, but the actual JSON object:"
              json_structure = json.dumps(self.template.output_json_structure, indent=2)
              formatted_prompt += f"{json_instruction}\n```json\n{json_structure}\n```"
          
          return formatted_prompt
  ```

### `llmcore/logger.py`

Sets up and manages logging throughout the library.

- **ColorFormatter**

  Custom formatter to add colors to log messages based on severity level.

  ```python
  class ColorFormatter(logging.Formatter):
      COLORS = {
          'DEBUG': '\033[94m',  # Blue
          'INFO': '\033[92m',   # Green
          'WARNING': '\033[93m',  # Yellow
          'ERROR': '\033[91m',  # Red
          'CRITICAL': '\033[95m',  # Magenta
          'RESET': '\033[0m'  # Reset color
      }
  
      def format(self, record):
          log_message = super().format(record)
          return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"
  ```

- **setup_logger**

  Sets up a logger with the custom formatter.

  ```python
  def setup_logger(name: str, level: Union[int, str] = logging.INFO):
      logger = logging.getLogger(name)
      logger.setLevel(level)
  
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setLevel(level)
  
      formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      console_handler.setFormatter(formatter)
  
      logger.addHandler(console_handler)
      return logger
  ```

**Notes:**
- These examples cover various functionalities such as LLM chaining, memory management, embedding and semantic search, prompt templating with parameter validation, and advanced chaining with memory support.
- Ensure you have the necessary environment variables set for the respective LLM providers before running these examples.
- Refer to the corresponding code files like `llmcore/core.py`, `llmcore/chain.py`, `llmcore/memory.py`, and others to understand the underlying implementations and for any further customizations.

If you need to modify other files or require additional examples, feel free to ask!