# LLMCore Documentation

## Overview

LLMCore is a Python package designed to facilitate the development and integration of Large Language Models (LLMs) into applications. It provides essential tools for managing models, prompts, embeddings, agents, chains, and more. This documentation covers the key components of the LLMCore package, including its modules, classes, and methods.

## Table of Contents

1. [Installation](#installation)
2. [Module Structure](#module-structure)
3. [Key Components](#key-components)
   - [Configuration](#configuration)
   - [Contracts](#contracts)
   - [Core](#core)
   - [Chain](#chain)
   - [Embeddings](#embeddings)
   - [Memory Management](#memory-management)
   - [Prompt Management](#prompt-management)
   - [Utilities](#utilities)
   - [Vector Databases](#vector-databases)
4. [Usage Examples](#usage-examples)
5. [Error Handling](#error-handling)

## Installation

To install LLMCore, you can use pip:

```bash
pip install llmcore
```

Make sure to have the required dependencies installed as specified in the `setup.py` file.

## Module Structure

The LLMCore package is organized into several modules, each serving a specific purpose:

- `llmcore/__init__.py`: Initializes the package and imports key components.
- `llmcore/config.py`: Contains configuration classes for LLM settings.
- `llmcore/contracts.py`: Defines data contracts for LLM responses and conversation management.
- `llmcore/core.py`: Implements the core functionality for interacting with LLMs.
- `llmcore/chain.py`: Provides classes for building and executing chains of LLM prompts.
- `llmcore/embeddings.py`: Manages embeddings for text and code.
- `llmcore/memory.py`: Implements memory management for storing and retrieving relevant information.
- `llmcore/prompt.py`: Manages prompt templates and their validation.
- `llmcore/utils.py`: Contains utility functions, such as cosine similarity calculations.
- `llmcore/vector_databases/`: Contains implementations for various vector databases.

## Key Components

### Configuration

The `LLMConfig` class in `llmcore/config.py` is used to configure the settings for LLM interactions.

```python
@dataclass
class LLMConfig:
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    response_format: Optional[Dict] = None
    json_response: bool = False
    json_instruction: Optional[str] = None
    vector_db_provider: Optional[str] = None
    vector_db_endpoint: Optional[str] = None
    vector_db_api_key: Optional[str] = None
```

### Contracts

The `contracts.py` module defines data structures for LLM responses and conversation management.

- `LLMResponse`: Represents a standard response from an LLM.
- `LLMStreamResponse`: Represents a streamed response from an LLM.
- `ConversationTurn`: Represents a single turn in a conversation.
- `Conversation`: Represents the history of a conversation.

### Core

The `core.py` module implements the main functionality for interacting with LLMs through various client adapters.

- `LLMClientAdapter`: An abstract base class for LLM client adapters.
- `APIClientAdapter`: A concrete implementation for making API calls to LLM providers.
- `OpenAIClientAdapter`, `AnthropicClientAdapter`, `GoogleGeminiClientAdapter`: Specific implementations for different LLM providers.

### Chain

The `chain.py` module provides classes for building and executing chains of LLM prompts.

- `LLMChain`: Represents a sequence of steps to execute with an LLM.
- `LLMChainStep`: Represents a single step in an LLM chain.
- `LLMChainBuilder`: A builder class for constructing LLM chains.

### Embeddings

The `embeddings.py` module manages the creation and retrieval of embeddings for text and code.

- `Embeddings`: A class for generating embeddings using a specified provider and model.
- `CodebaseEmbeddings`: A specialized class for managing code snippets and their embeddings.

### Memory Management

The `memory.py` module implements memory management for storing and retrieving relevant information.

- `MemoryManager`: Manages the storage and retrieval of memories based on vector embeddings.
- `Vector`: A class representing a mathematical vector for similarity calculations.

### Prompt Management

The `prompt.py` module manages prompt templates and their validation.

- `PromptTemplate`: Represents a template for creating prompts with placeholders.
- `Prompt`: Represents a formatted prompt ready for submission to an LLM.

### Utilities

The `utils.py` module contains utility functions, such as cosine similarity calculations.

- `cosine_similarity`: Computes the cosine similarity between two vectors.

### Vector Databases

The `vector_databases` module contains implementations for various vector databases.

- `VectorDatabase`: An abstract base class for vector database implementations.
- `PineconeDatabase`: A concrete implementation for Pinecone vector database.
- `ChromaDatabase`: A concrete implementation for Chroma vector database.

## Usage Examples

### Basic LLM Interaction

```python
from llmcore import LLM, LLMConfig

# Initialize LLM
llm = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.7))

# Send a prompt
response = llm.send_input("What is the capital of France?")
print(response)  # Output: "The capital of France is Paris."
```

### Building an LLM Chain

```python
from llmcore import LLMChainBuilder, PromptTemplate

# Create prompt templates
template1 = PromptTemplate("What is the capital of {{country}}?", required_params={"country": str})
template2 = PromptTemplate("Provide a brief history of {{country}}.", required_params={"country": str})

# Build the chain
chain = (
    LLMChainBuilder()
    .add_step(template=template1.template, output_key="capital", required_params=template1.required_params)
    .add_step(template=template2.template, output_key="history", required_params=template2.required_params)
    .build()
)

# Execute the chain
result = chain.execute({"country": "France"})
print(result)  # Output: {'capital': 'Paris', 'history': 'France has a rich history...'}
```

## Error Handling

LLMCore provides custom exceptions for error handling:

- `LLMAPIError`: Raised for API-related errors.
- `LLMJSONParseError`: Raised for JSON parsing errors.
- `LLMPromptError`: Raised for prompt-related errors.
- `LLMNetworkError`: Raised for network-related errors.

You can catch these exceptions to handle errors gracefully in your application.

```python
try:
    response = llm.send_input("What is the capital of France?")
except LLMAPIError as e:
    print(f"API error occurred: {e}")
except LLMJSONParseError as e:
    print(f"JSON parsing error: {e}")
```

## Conclusion

LLMCore is a powerful package for integrating and managing LLMs in your applications. With its modular design and comprehensive features, it simplifies the process of working with language models, embeddings, and memory management. For further details, refer to the individual module documentation and usage examples provided above.