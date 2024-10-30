from typing import Dict, Any, Union, AsyncGenerator, Generator, List
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urljoin
from functools import lru_cache
import aiohttp
import asyncio
import typing
import json
import os
import re

from llmcore.prompt import Prompt, PromptTemplate
from llmcore.config import LLMConfig
from llmcore.contracts import ConversationTurn
from llmcore.memory import MemoryManager
from llmcore.embeddings import Embeddings

class LLMAPIError(Exception):
    pass

class LLMJSONParseError(Exception):
    pass

class LLMPromptError(Exception):
    pass

class LLMNetworkError(Exception):
    pass

@dataclass
class RelevantMemory:
    content: str
    score: float

class APIEndpoints:
    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com/v1"
    GOOGLE_GEMINI = "https://generativelanguage.googleapis.com/v1beta"

class LLMClientAdapter(ABC):
    @abstractmethod
    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        pass

    @abstractmethod
    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        pass

class APIClientAdapter(LLMClientAdapter):
    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 3, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    # NOTE: This is for debugging purposes only. It allows you to copy and paste the
    #       cURL command into your terminal to test the API call in case you're having
    #       trouble getting the request to work.
    def _generate_curl_command(self, url: str, data: Dict, headers: Dict) -> str:
        header_args = ' '.join([f"-H '{k}: {v}'" for k, v in headers.items()])
        data_arg = f"-d '{json.dumps(data)}'"
        return f"curl -X POST {header_args} {data_arg} '{url}'"

    async def _make_request(self, endpoint: str, data: Dict, headers: Dict) -> Dict:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        # curl_command = self._generate_curl_command(url, data, headers)
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=data, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                error_message = f"API request failed: {str(e)}"
                try:
                    error_content = e.message
                    error_message += f"\nResponse content: {error_content}"
                except:
                    error_message += "\nCouldn't retrieve response content." 
                raise LLMAPIError(error_message)
            except aiohttp.ClientError as e:
                error_message = f"Network error occurred: {str(e)}"
                raise LLMNetworkError(error_message)

    async def _stream_request(self, endpoint: str, data: Dict, headers: Dict) -> AsyncGenerator[Dict, None]:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                if self.model.startswith("claude-"):
                    async for chunk in self._stream_anthropic(response):
                        yield chunk
                elif self.model.startswith("gpt-"):
                    async for chunk in self._stream_openai(response):
                        yield chunk
                elif self.model.startswith("gemini-"):
                    async for chunk in self._stream_google(response):
                        yield chunk
                else:
                    async for chunk in self._stream_default(response):
                        yield chunk

    async def _stream_anthropic(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        parsed_json = json.loads(json_str)
                        yield parsed_json
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

    async def _stream_openai(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(json_str)
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

    async def _stream_google(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(json_str)
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

    async def _stream_default(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON


    @abstractmethod
    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        pass

    @abstractmethod
    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        pass

class OpenAIClientAdapter(APIClientAdapter):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, APIEndpoints.OPENAI, model)

    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p
        }

        if config.response_format:
            data["response_format"] = config.response_format

        response = await self._make_request("/chat/completions", data, headers)
        return response["choices"][0]["message"]["content"]

    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        try:
            async for chunk in self._stream_request("/chat/completions", data, headers):
                if "choices" in chunk and chunk["choices"]:
                    if "content" in chunk["choices"][0]["delta"]:
                        yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            raise e

class AnthropicClientAdapter(APIClientAdapter):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, APIEndpoints.ANTHROPIC, model)

    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        try:
            response = await self._make_request("/messages", data, headers)
            return response["content"][0]["text"]
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                try:
                    error_content = json.loads(e.message)
                    if error_content.get("type") == "error" and error_content.get("error", {}).get("type") == "invalid_request_error":
                        error_message = error_content["error"]["message"]
                        if "credit balance is too low" in error_message:
                            raise LLMAPIError(f"Anthropic API request failed: Insufficient credits. {error_message}")
                except json.JSONDecodeError as json_err:
                    pass
            raise LLMAPIError(f"Anthropic API request failed: {e}")
        except Exception as e:
            raise LLMAPIError(f"Anthropic API request failed: {e}")

    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        try:
            async for chunk in self._stream_request("/messages", data, headers):
                if "delta" in chunk and "text" in chunk["delta"]:
                    yield chunk["delta"]["text"]
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                try:
                    error_content = json.loads(e.message)
                    if error_content.get("type") == "error" and error_content.get("error", {}).get("type") == "invalid_request_error":
                        error_message = error_content["error"]["message"]
                        if "credit balance is too low" in error_message:
                            raise LLMAPIError(f"Anthropic API request failed: Insufficient credits. {error_message}")
                except json.JSONDecodeError as json_err:
                    pass
            raise LLMAPIError(f"Anthropic API request failed: {e}")
        except Exception as e:
            raise LLMAPIError(f"Anthropic API request failed: {e}")

class GoogleGeminiClientAdapter(APIClientAdapter):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, APIEndpoints.GOOGLE_GEMINI, model)

    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "maxOutputTokens": config.max_tokens,
            }
        }
        endpoint = f"/models/{self.model}:generateContent?key={self.api_key}"
        response = await self._make_request(endpoint, data, headers)
        return response["candidates"][0]["content"]["parts"][0]["text"]

    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "maxOutputTokens": config.max_tokens,
            }
        }
        endpoint = f"/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        async for chunk in self._stream_request(endpoint, data, headers):
            if "candidates" in chunk and chunk["candidates"]:
                content = chunk["candidates"][0]["content"]["parts"][0]["text"]
                if content:
                    yield content

class LLM:
    JSON_ENSURE_RESPONSE = "\n\nPlease ensure your entire response is valid JSON."
    
    def __init__(self, provider: str, model: str, config: LLMConfig = LLMConfig()):
        self.provider = provider
        self.model = model
        self.config = config
        self.client = self.load_model()
        self.embeddings = Embeddings(provider="openai", model="text-embedding-3-small")
        self.memory_manager = MemoryManager(config = config, capacity = 32000)

    def load_model(self):
        api_provider = self.provider.lower() if self.provider != "google" else "gemini"
        api_key = os.environ.get(f"{api_provider.upper()}_API_KEY")
        
        if not api_key:
            # Check user-level environment variables
            try:
                from dotenv import load_dotenv
                load_dotenv()  # This loads the .env file from the user's home directory
                api_key = os.environ.get(f"{api_provider.upper()}_API_KEY")
            except ImportError:
                pass  # dotenv is not installed, continue with system-level check
        
        if not api_key:            
            raise ValueError(f"API key ({api_provider.upper()}_API_KEY) for {self.provider} not found in system or user environment variables")

        if self.provider == "openai":
            return OpenAIClientAdapter(api_key, self.model)
        elif self.provider == "anthropic":
            return AnthropicClientAdapter(api_key, self.model)
        elif self.provider == "gemini" or self.provider == "google":
            return GoogleGeminiClientAdapter(api_key, self.model)
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")

    def send_input(self, prompt: Prompt, parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        formatted_prompt: str = prompt.format()
        
        # Check if we're in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If we're not in an event loop, run the async method and return the result
            return asyncio.run(self._send_input_async(formatted_prompt, parse_json, prompt.template.output_json_structure))

        # If we are in an event loop, we need to run our coroutine in a separate thread
        with ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run, 
                self._send_input_async(formatted_prompt, parse_json, prompt.template.output_json_structure)
            )
            return future.result()

    def stream_input(self, prompt: Prompt, parse_json: bool = False) -> Generator[Union[str, Dict[str, Any]], None, None]:
        async def async_generator():
            formatted_prompt: str = prompt.format()
            async for chunk in self.stream_input_async(formatted_prompt, parse_json=parse_json, output_json_structure=prompt.template.output_json_structure):
                yield chunk

        return self._async_to_sync_generator(async_generator())

    async def send_input_async(self, prompt: Prompt, parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        formatted_prompt = prompt.format()
        return await self._send_input_async(formatted_prompt, parse_json, prompt.template.output_json_structure)

    async def stream_input_async(self, prompt: Prompt, parse_json: bool = False) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        formatted_prompt = prompt.format()
        self._configure_json_mode(prompt.template.output_json_structure is not None)

        try:
            accumulated_json = ""
            async for chunk in self.client.stream_prompt(formatted_prompt, self.config):
                if parse_json:
                    yield chunk  # Stream the raw chunk
                    accumulated_json += chunk
                    try:
                        parsed_json = json.loads(accumulated_json)
                        yield self._extract_fields(parsed_json, prompt.template.output_json_structure)
                        accumulated_json = ""  # Reset after successful parse
                    except json.JSONDecodeError:
                        # Continue accumulating if it's not a complete JSON yet
                        continue
                else:
                    yield chunk

        except aiohttp.ClientError as e:
            raise LLMNetworkError(f"Network error occurred while streaming from LLM: {str(e)}")
        except json.JSONDecodeError as e:
            raise LLMJSONParseError(f"Failed to parse JSON from LLM response: {str(e)}")
        except Exception as e:
            raise LLMAPIError(f"Unexpected error occurred while streaming from LLM: {str(e)}")

    async def send_input_with_history(self, prompt: Prompt, history: List[ConversationTurn], parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        formatted_prompt = prompt.format()
        full_prompt = self._build_prompt_with_history(formatted_prompt, history)
        return await self._send_input_async(full_prompt, parse_json, prompt.template.output_json_structure)

    def _build_prompt_with_history(self, current_prompt: str, history: List[ConversationTurn]) -> str:
        conversation = "\n".join([f"{turn.role}: {turn.content}" for turn in history])
        return f"{conversation}\n\nHuman: {current_prompt}\nAI:"

    @lru_cache(maxsize=1)
    def _get_fast_llm(self):
        fast_config = LLMConfig(temperature=0.3, max_tokens=100, json_response=True)
        return LLM("openai", "gpt-4o-mini", config=fast_config)

    async def _format_memory_for_storage(self, prompt: str, response: str) -> str:
        fast_llm = self._get_fast_llm()
        format_prompt = PromptTemplate(
            template="""
Analyze this human-AI interaction to extract comprehensive memories for future use.
----------------------------------------
Human: {{prompt}}
----------------------------------------
AI: {{response}}
----------------------------------------

Create a detailed, reusable memory capturing all relevant information from the interaction. Do not omit any significant details. The memory should be thorough enough to provide a complete understanding of the interaction when reviewed later.
            """,
            required_params={"prompt": str, "response": str},
            output_json_structure={"memory": str}
        )
        result = await fast_llm.send_input_async(format_prompt.create_prompt(prompt=prompt, response=response), parse_json=True)
        if isinstance(result, dict):
            return result["memory"]
        else:
            return result

    async def send_input_with_memory(self, prompt: Prompt, parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        formatted_prompt = prompt.format()

        # Convert the prompt to a vector using embeddings
        prompt_vector = None
        try:
            prompt_vector = await self.embeddings.embed_async(formatted_prompt)
        except Exception as e:
            print(f"Error creating embedding for prompt: {e}")

        # Use the vector to get relevant memories
        relevant_memories = []
        if prompt_vector is not None:
            try:
                relevant_memories = await self.memory_manager.get_relevant_memories(prompt_vector, k=3)
            except Exception as e:
                print(f"Error retrieving relevant memories: {e}")

        memory_context = "\n".join([f"Memory (score: {mem.score:.2f}): {mem.content}" for mem in relevant_memories])
        full_prompt = f"""Relevant Memories:
{memory_context}

Use the above memories as a starting point, but also incorporate your own knowledge and understanding to provide a comprehensive response.

Prompt: {formatted_prompt}

Response:"""

        try:
            response = await self._send_input_async(full_prompt, parse_json, prompt.template.output_json_structure)
        except Exception as e:
            print(f"Error sending input async: {e}")
            raise

        # Format the memory using a fast LLM
        try:
            formatted_memory = await self._format_memory_for_storage(formatted_prompt, response['response'])
        except Exception as e:
            print(f"Error formatting memory for storage: {e}")
            return response

        vector = None
        try:
            vector = await self.embeddings.embed_async(formatted_memory)
        except Exception as e:
            print(f"Error creating embedding for formatted memory: {e}")

        if vector is not None:
            try:
                # Add the new interaction to memory only if we have a valid vector
                await self.memory_manager.add_memory({
                    "content": formatted_memory,
                    "vector": vector
                })
            except Exception as e:
                print(f"Error adding memory: {e}")

        return response

    async def _send_input_async(self, formatted_prompt: str, parse_json: bool, output_json_structure: Dict[str, Any] = None) -> Union[str, Dict[str, Any]]:
        self._configure_json_mode(output_json_structure is not None)

        try:
            response = await self.client.send_prompt(formatted_prompt, self.config)
            if parse_json:
                try:
                    parsed_response = await self._parse_json_response(response, output_json_structure)
                    if not isinstance(parsed_response, dict):
                        raise LLMJSONParseError("LLM did not return a valid JSON object")
                    return parsed_response
                except json.JSONDecodeError as e:
                    raise LLMJSONParseError(f"Failed to parse JSON from the LLM response: {str(e)}")
            return response
        except LLMJSONParseError as e:
            raise
        except aiohttp.ClientError as e:
            raise LLMNetworkError(f"Network error occurred while sending prompt to LLM: {str(e)}")
        except Exception as e:
            raise LLMAPIError(f"Unexpected error occurred while sending prompt to LLM: {str(e)}")

    async def _parse_json_response(self, response: str, expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_json = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it's not already in JSON format
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1))
                    return self._extract_fields(parsed_json, expected_structure)
                except json.JSONDecodeError:
                    pass

            llm_prompt = (
                "Extract and return only the JSON object from the following text. "
                "Ensure the output is valid JSON without any additional text.\n\n"
                f"Text: \"{response}\""
            )

            try:
                extracted_json = await self.client.send_prompt(llm_prompt, self.config)
                return extracted_json
            except Exception as e:
                raise LLMJSONParseError("Failed to extract JSON from the LLM response using LLM")

        return self._extract_fields(parsed_json, expected_structure)

    def _extract_fields(self, parsed_json: Dict[str, Any], expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, expected_type in expected_structure.items():
            if key not in parsed_json:
                raise LLMJSONParseError(f"Missing expected field: {key}")
            
            value = parsed_json[key]
            if isinstance(expected_type, str):
                result[key] = self._validate_type(value, expected_type, key)
            elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                valid_types = [t for t in expected_type.__args__ if not isinstance(t, type(typing.Any))]
                if not any(self._is_valid_type(value, t) for t in valid_types):
                    raise LLMJSONParseError(f"Field {key} should be one of {valid_types}")
                result[key] = value
            else:
                result[key] = self._validate_type(value, expected_type, key)
        
        return result

    def _is_valid_type(self, value, expected_type):
        if hasattr(expected_type, '__origin__'):
            if expected_type.__origin__ is dict:
                return isinstance(value, dict)
            elif expected_type.__origin__ is list:
                return isinstance(value, list)
        return isinstance(value, expected_type)

    def _validate_type(
        self, value: Any, expected_type: Union[str, type, Dict[str, Any]], field_name: str
    ) -> Any:
        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                raise LLMJSONParseError(f"Field {field_name} should be a dictionary")
            return self._extract_fields(value, expected_type)

        if isinstance(expected_type, str):
            if expected_type == 'int':
                if not isinstance(value, (int, float)):
                    raise LLMJSONParseError(f"Field {field_name} should be an integer or float")
                return int(value)
            elif expected_type == 'float':
                if not isinstance(value, (int, float)):
                    raise LLMJSONParseError(f"Field {field_name} should be a float")
                return float(value)
            elif expected_type == 'str':
                if not isinstance(value, str):
                    raise LLMJSONParseError(f"Field {field_name} should be a string")
                return value
            elif expected_type == 'bool':
                if not isinstance(value, bool):
                    raise LLMJSONParseError(f"Field {field_name} should be a boolean")
                return value
            elif expected_type.startswith('dict['):
                if not isinstance(value, dict):
                    raise LLMJSONParseError(f"Field {field_name} should be a dictionary")
                return value
            elif expected_type.startswith('list['):
                if not isinstance(value, list):
                    raise LLMJSONParseError(f"Field {field_name} should be a list")
                return value
            elif expected_type.startswith('Union['):
                # Handle Union types
                union_types = expected_type[6:-1].split(',')
                for union_type in union_types:
                    try:
                        return self._validate_type(value, union_type.strip(), field_name)
                    except LLMJSONParseError:
                        continue
                raise LLMJSONParseError(f"Field {field_name} does not match any type in {expected_type}")
            else:
                raise LLMJSONParseError(f"Unsupported type for field {field_name}: {expected_type}")

        elif isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                raise LLMJSONParseError(f"Field {field_name} should be of type {expected_type.__name__}")
            return value

        elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            # Handle typing.Union types
            for union_type in expected_type.__args__:
                try:
                    return self._validate_type(value, union_type, field_name)
                except LLMJSONParseError:
                    continue
            raise LLMJSONParseError(f"Field {field_name} does not match any type in {expected_type}")

        else:
            raise LLMJSONParseError(f"Unsupported type for field {field_name}: {expected_type}")
        
    def _configure_json_mode(self, json_response: bool):
        if json_response:
            if self.provider == "openai" and "gpt-4" in self.model:
                self.config.response_format = {"type": "json_object"}
            elif self.provider in ["anthropic", "google"]:
                self.config.json_response = True
                if "claude-3" in self.model or "gemini" in self.model:
                    self.config.json_instruction = self.JSON_ENSURE_RESPONSE
        else:
            self.config.response_format = None

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid configuration option: {key}")

    @staticmethod
    def _async_to_sync_generator(async_gen):
        agen = async_gen.__aiter__()
        loop = asyncio.get_event_loop()
        try:
            while True:
                yield loop.run_until_complete(agen.__anext__())
        except StopAsyncIteration:
            pass